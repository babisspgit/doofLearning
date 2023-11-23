import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from transformers import BertModel, BertTokenizer

from src.models.models import Transformers_Bert

from src.data.make_dataset import DatasetRecipes

from tqdm import tqdm
from pathlib import Path
import random
import numpy as np


import hydra
from omegaconf import OmegaConf

import wandb


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(
    config_path="config/ViT_BertConfig",
    config_name="default_config.yaml",
    version_base=None,
)
def main(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Unpack hparams
    seed = config.seed

    set_seed(seed=seed)

    lr = config.lr
    batch_size = config.batch_size
    n_epochs = config.n_epochs
    save_per_n_epochs = config.save_per_n_epochs

    scheduler_step = config.scheduler_step
    scheduler_gamma = config.scheduler_gamma

    # data path(s)
    data_path = Path(config.data_path)
    save_path = Path(config.model_save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    save_model_path = save_path / "ViT_Bert_Transf.pt"

    # Unpack experiment specific params
    hparams = config["_group_"]  # wtf is this __group__ ?

    embed_dim = hparams.embed_dim

    # ViT
    image_dims = hparams.vit.image_dims
    patch_dims = hparams.vit.patch_dims
    num_heads_vit = hparams.vit.num_heads
    num_blocks_vit = hparams.vit.num_blocks

    wandb.init(project=f"ViT_Text_Transf")
    wandb.config = {
        "learning_rate": lr,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "text": "Bert",
    }
    # ViT
    image_dims = [image_dims, image_dims]
    patch_dims = (patch_dims, patch_dims)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_dims),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    columns = ["Title"]

    train_set = DatasetRecipes(
        data_path / "train", columns=columns, transformations=train_transform
    )
    val_set = DatasetRecipes(
        data_path / "validation", columns=columns, transformations=train_transform
    )

    MAX_SEQ_LEN = 512  # Fixed for Bert
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def collate_batch(batch):
        img_list, text_list, attention_masks_list = [], [], []
        for img, _text in batch:
            # Protect against larger sequencies
            # -2 to account for [CLS] and [SEP] tokens
            if len(_text) > MAX_SEQ_LEN - 2:
                _text = _text[: MAX_SEQ_LEN - 2]

            processed_text = tokenizer.tokenize(_text)

            # Add special characters, like in
            # https://jalammar.github.io/illustrated-bert/
            processed_text = ["[CLS]"] + processed_text + ["[SEP]"]
            pads = ["[PAD]" for _ in range(abs(MAX_SEQ_LEN - len(processed_text)))]
            processed_text = processed_text + pads

            attention_mask = [0 if i == "[PAD]" else 1 for i in processed_text]

            processed_text = tokenizer.convert_tokens_to_ids(processed_text)

            processed_text = torch.tensor(processed_text, dtype=torch.int64)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)

            text_list.append(processed_text.unsqueeze(0))
            attention_masks_list.append(attention_mask.unsqueeze(0))

            # Since the batching is manual, in this fcn, I need to add batch dim
            img_list.append(img.unsqueeze(0))

        return (
            torch.cat(img_list, axis=0).to(device),
            torch.cat(text_list, axis=0).to(device),
            torch.cat(attention_masks_list, axis=0).to(device),
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_batch, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=collate_batch, shuffle=False
    )

    vit_options = {
        "img_dims": image_dims,
        "channels": 3,
        "patch_sizes": patch_dims,
        "embed_dim": embed_dim,
        "num_heads": num_heads_vit,
        "num_layers": num_blocks_vit,
    }

    # model = BertModel.from_pretrained("bert-base-uncased")
    model = Transformers_Bert(vit_options)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    text_loss = nn.CrossEntropyLoss()
    image_loss = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_accuracy = 0
        pbar = tqdm(train_loader)

        for batch in pbar:
            img, in_idx, att_masks = batch

            # Use contrastive loss
            labels = torch.arange(img.shape[0])  # should be eq to batch_size
            labels = labels.to(device)

            _, _, logits_per_text, logits_per_image = model(
                img, in_idx, attention_mask=att_masks
            )

            loss_img = image_loss(logits_per_image, labels)
            loss_text = text_loss(
                logits_per_text, labels
            )  # maybe logits transpose? should not matter at all

            loss = (loss_img + loss_text) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            probs = torch.nn.functional.softmax(logits_per_text, dim=1)
            preds = torch.argmax(probs, dim=-1)

            accuracy = (preds == labels).sum().item()

            train_accuracy += accuracy

            pbar.set_description(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        pbar = tqdm(val_loader)
        val_accuracy = 0
        val_loss = 0
        for batch in pbar:
            with torch.no_grad():
                img, in_idx, att_masks = batch

                # Use contrastive loss
                labels = torch.arange(img.shape[0])  # should be eq to batch_size
                labels = labels.to(device)

                _, _, logits_per_text, logits_per_image = model(
                    img, in_idx, attention_mask=att_masks
                )

                loss_img = image_loss(logits_per_image, labels)
                loss_text = text_loss(
                    logits_per_text, labels
                )  # maybe logits transpose? should not matter at all

                loss = (loss_img + loss_text) / 2.0

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
                "training_loss": train_loss / len(train_set) * batch_size,
                "training_accuracy": train_accuracy / len(train_set),
                "validation_loss": val_loss / len(val_set) * batch_size,
                "validation_accuracy": val_accuracy / len(val_set),
            }
        )

        # Save the model state dict
        if epoch % save_per_n_epochs == 0:
            torch.save(
                model.state_dict(),
                save_model_path,
            )
            print("Saved model")


if __name__ == "__main__":
    main()
