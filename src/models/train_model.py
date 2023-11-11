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
from src.models.models import (
    TransformersSingleTextModel,
    Transformers_Bert,
    VGG_SingleTextModel,
    VGGpre_SingleTextModel,
    VGG_Bert,
    VGGpre_Bert,
)

from src.utils.vocab_build import get_vocab, tokenizer

# from transformers import BertTokenizer

# from src.models.losses import ConstrastiveLoss, ClipSymmetricLoss

MAX_SEQ_LEN = 512  # Maximum number of tokens per text input
VOCAB_SIZE = 50000


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(data_path, n_epochs=20, batch_size=4, seed=0, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_seed(
        seed=seed
    )  # To make sure, because the same seed will be used for test set in another file

    train_path = data_path / "train"

    train_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),  # like CLIP
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = DatasetRecipes(train_path, transformations=train_transform)

    # Use a custom made vocabulary based on the text we have. See fcn for ref.

    # btokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    vocab = get_vocab(train_dataset, tokenizer=tokenizer)  # BertTokenizer
    # BertTokenizer

    # Pipeline
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

    def collate_batch(batch):
        img_list, text_list = [], []
        for img, _text in batch:
            if len(_text) > MAX_SEQ_LEN:
                _text = _text[:MAX_SEQ_LEN]
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            img_list.append(img.unsqueeze(0))
        text_list[0] = nn.ConstantPad1d((0, MAX_SEQ_LEN - text_list[0].shape[0]), 0)(
            text_list[0]
        )
        padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        return (torch.cat(img_list, axis=0).to(device), padded_text_list.to(device))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )

    img_size = train_dataset[0][0].shape[-2:]
    patches_size = (32, 32)

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

    # textBert_transf_options = {
    #    "num_heads": 1,
    #    "num_blocks": 3,
    #    "embed_dims": 128,
    #    "projection_dims": 128,
    #    "vocab_size": VOCAB_SIZE,
    #    "max_seq_len": MAX_SEQ_LEN,
    # }

    model = Transformers_Bert(vit_options, text_transf_options)
    # model = VGGpre_SingleTextModel(vit_options, text_transf_options)  ## works
    # model = VGG_SingleTextModel(vit_options, text_transf_options)      ## works
    # model = TransformersSingleTextModel(vit_options, text_transf_options)
    model.to(device)

    # freeze ????
    # if (model == VGGpre_Bert) or (model == VGGpre_SingleTextModel):
    #    for param in model.img_model.vgg.features.parameters():
    #         param.requires_grad = False

    optim = torch.optim.AdamW(model.parameters(), lr=lr)  # Should we add weight decay?

    text_loss = nn.CrossEntropyLoss()
    image_loss = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader)
        for data in pbar:
            optim.zero_grad()

            img, text = data

            curr_batch_size = img.shape[0]

            logits_per_text, logits_per_image = model(img, text)
            labels = torch.arange(curr_batch_size).to(device)

            batch_text_loss = text_loss(logits_per_text, labels)
            batch_image_loss = image_loss(logits_per_image, labels)

            loss = (batch_image_loss + batch_text_loss) / 2.0

            loss.backward()

            optim.step()

        logger.info(f"Epoch {epoch} loss: {loss}")

        # Save the model state dict


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    options = {
        "data_path": Path(r"data/processed"),
        "batch_size": 32,
        "n_epochs": 20,
        "seed": 0,
        "lr": 1e-4,
    }

    main(**options)
