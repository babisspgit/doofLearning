import logging
import random

from pathlib import Path

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from src.data.make_dataset import DatasetRecipes
from src.models.models import TransformersSingleTextModel
from src.utils.vocab_build import get_vocab, CustomTokenizer


import wandb

import hydra
from omegaconf import OmegaConf


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(
    config_path="config/ViT_TxtTransfConfig",
    config_name="default_config.yaml",
    version_base=None,
)
def main(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Unpack hparams
    hparams = config["_group_"]  # wtf is this __group__ ?

    seed = hparams.seed
    lr = hparams.lr
    embed_dim = hparams.embed_dim
    batch_size = hparams.batch_size
    n_epochs = hparams.n_epochs
    save_per_n_epochs = hparams.save_per_n_epochs

    # data path
    data_path = Path(hparams.data_path)

    # ViT
    image_dims = hparams.vit.image_dims
    patch_dims = hparams.vit.patch_dims
    num_heads_vit = hparams.vit.num_heads
    num_blocks_vit = hparams.vit.num_blocks

    # Text Transformer
    num_heads_text = hparams.text_transf.num_heads
    num_blocks_text = hparams.text_transf.num_blocks

    wandb.init(project=f"ViT_Text_Transf")
    wandb.config = {
        "learning_rate": lr,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "vit": hparams.vit,
        "text_transf": hparams.text_transf,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_seed(
        seed=seed
    )  # To make sure, because the same seed will be used for test set in another file

    train_path = data_path / "train"
    validation_path = data_path / "validation"

    train_transform = transforms.Compose(
        [
            transforms.Resize([image_dims, image_dims]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    columns = []

    train_dataset = DatasetRecipes(
        train_path, columns=columns, transformations=train_transform
    )
    val_dataset = DatasetRecipes(
        validation_path, columns=columns, transformations=train_transform
    )

    # Use a custom made vocabulary based on the text we have. See fcn for ref.
    tokenizer = CustomTokenizer()

    vocab_, MAX_SEQ_LEN = get_vocab(train_dataset, tokenizer=tokenizer.tokenize)

    VOCAB_SIZE = len(vocab_)

    # print(vocab_["<bos>"], vocab_["<eos>"], vocab_["<pad>"], vocab_["<unk>"])

    # print(vocab_["malakas"], vocab_["<unk>"])

    # Pipeline
    text_pipeline = lambda x: [vocab_[token] for token in tokenizer.tokenize(x)]

    def collate_batch(batch):
        img_list, text_list = [], []
        for img, _text in batch:
            # Protect against larger sequencies
            # -2 to account for BOS and EOS tokens
            if len(_text) > MAX_SEQ_LEN - 2:
                _text = _text[: MAX_SEQ_LEN - 2]

            processed_text = text_pipeline(_text)

            # Insert bos token and eos token

            processed_text = [vocab_["<bos>"]] + processed_text + [vocab_["<eos>"]]

            # pad
            # NOTE: I am padding all the way from the right ONLY, torch uses a padding method from both sides
            pad = [vocab_["<pad>"]] * (MAX_SEQ_LEN - len(processed_text))

            processed_text = processed_text + pad

            processed_text = torch.tensor(processed_text, dtype=torch.int64)

            # nn.ConstantPad1d((0, MAX_SEQ_LEN - text_list[0].shape[0]), 0)(text_list[0])

            # Prepare text and image batch
            # processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text.unsqueeze(0))

            # Since the batching is manual, in this fcn, I need to add batch dim
            img_list.append(img.unsqueeze(0))

        return (
            torch.cat(img_list, axis=0).to(device),
            torch.cat(text_list, axis=0).to(device),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    img_size = train_dataset[0][0].shape[-2:]
    patches_size = (patch_dims, patch_dims)

    vit_options = {
        "img_dims": img_size,
        "channels": 3,
        "patch_sizes": patches_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads_vit,
        "num_layers": num_blocks_vit,
    }

    text_transf_options = {
        "num_heads": num_heads_text,
        "num_blocks": num_blocks_text,
        "embed_dims": embed_dim,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
    }

    model = TransformersSingleTextModel(vit_options, text_transf_options)
    model.to(device)

    # optim = torch.optim.AdamW(model.parameters(), lr=lr)  # Should we add weight decay?

    optim = torch.optim.Adam(model.parameters(), lr=lr)  # weight decay: L2 penalty 1e-5
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

    text_loss = nn.CrossEntropyLoss()
    image_loss = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_accuracy = 0
        pbar = tqdm(train_loader)
        for data in pbar:
            optim.zero_grad()

            img, text = data

            curr_batch_size = img.shape[0]
            labels = torch.arange(curr_batch_size).to(device)

            logits_per_text, logits_per_image, _, _ = model(img, text)

            batch_text_loss = text_loss(logits_per_text, labels)
            batch_image_loss = image_loss(logits_per_image, labels)

            loss = (batch_image_loss + batch_text_loss) / 2.0

            loss.backward()

            optim.step()

            train_loss += loss.item()

            probs = torch.nn.functional.softmax(logits_per_text, dim=1)
            preds = torch.argmax(probs, dim=-1)

            accuracy = (preds == labels).sum().item()

            train_accuracy += accuracy

            pbar.set_description(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}")

        # Step scheduler
        # scheduler.step()

        # Validation
        model.eval()
        pbar = tqdm(val_loader)
        val_accuracy = 0
        val_loss = 0
        for batch in pbar:
            with torch.no_grad():
                img, text = batch
                curr_batch_size = img.shape[0]
                labels = torch.arange(curr_batch_size).to(device)

                logits_per_text, logits_per_image, _, _ = model(img, text)
                labels = torch.arange(curr_batch_size).to(device)

                batch_text_loss = text_loss(logits_per_text, labels)
                batch_image_loss = image_loss(logits_per_image, labels)

                loss = (batch_image_loss + batch_text_loss) / 2.0

                val_loss += loss.item()

                probs = torch.nn.functional.softmax(logits_per_text, dim=1)
                preds = torch.argmax(probs, dim=-1)

                accuracy = (preds == labels).sum().item()

                val_accuracy += accuracy

                pbar.set_description(
                    f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}"
                )

        # Log training epoch data
        wandb.log(
            {
                "training_loss": train_loss / len(train_dataset) * batch_size,
                "training_accuracy": train_accuracy / len(train_dataset),
                "validation_loss": val_loss / len(val_dataset) * batch_size,
                "validation_accuracy": val_accuracy / len(val_dataset),
            }
        )

        # Save the model state dict
        if epoch % save_per_n_epochs == 0:
            torch.save(
                model.state_dict(),
                Path(
                    "models/ViT_Text_Tranf_lr_{lr}_emb_{embed_dim}_heads_{num_heads}_n_blocks_{n_blocks}.pt"
                ),
            )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    main()
