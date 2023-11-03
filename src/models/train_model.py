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

from src.utils.vocab_build import get_vocab, tokenizer

from transformers import BertTokenizer

from src.models.losses import ConstrastiveLoss, ClipSymmetricLoss

MAX_SEQ_LEN = 2500  # Maximum number of tokens per text input
VOCAB_SIZE = 50000


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(data_path, n_epochs=20, batch_size=16, seed=0, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_seed(
        seed=seed
    )  # To make sure, because the same seed will be used for test set in another file

    train_path = data_path / "train"

    train_transform = transforms.Compose(
        [
            transforms.Resize([169, 169]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = DatasetRecipes(train_path, transformations=train_transform)

    # Use a custom made vocabulary based on the text we have. See fcn for ref.
    vocab = get_vocab(train_dataset, tokenizer=tokenizer)  # BertTokenizer

    # Pipeline
    device = torch.device("cpu")

    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

    def collate_batch(batch):
        img_list, text_list, lengths = [], [], []
        for img, _text in batch:
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            img_list.append(img.unsqueeze(0))
            lengths.append(processed_text.size(0))
        lengths = torch.tensor(lengths)
        text_list[0] = nn.ConstantPad1d((0, MAX_SEQ_LEN - text_list[0].shape[0]), 0)(
            text_list[0]
        )
        padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        return (
            torch.cat(img_list, axis=0).to(device),
            padded_text_list.to(device),
            lengths.to(device),  # might not actually need this
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    img_size = train_dataset[0][0].shape[-2:]
    patches_size = (13, 13)

    vit_options = {
        "img_dims": img_size,
        "channels": 3,
        "patch_sizes": patches_size,
        "embed_dim": 128,
        "projection_dims": 128,
        "num_heads": 2,
        "num_layers": 4,
    }

    text_transf_options = {
        "num_heads": 1,
        "num_blocks": 3,
        "embed_dims": 128,
        "projection_dims": 128,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
    }

    model = TransformersSingleTextModel(vit_options, text_transf_options)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)  # Should we add weight decay?
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim,
    # )  # or stepLR

    loss_fn = ConstrastiveLoss(m=1.0)

    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_loss = 0
        for data in train_loader:
            img, text, _ = data  # from collate_fn they are on device

            # img_batch_features, text_batch_features = model(img, text)

            # optim.zero_grad()
            # loss = loss_fn(img_batch_features, text_batch_features)
            # loss.backward()

            # optim.step()

            # epoch_loss += loss.detach().numpy()

            # return

        logger.info(f"Epoch {epoch} loss: {epoch/batch_size}")

        # Save the model state dict

        # scheduler.step()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    options = {
        "data_path": Path(r"data/processed"),
        "batch_size": 1,
        "n_epochs": 20,
        "seed": 0,
        "lr": 1e-4,
    }

    main(**options)
