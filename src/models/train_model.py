import logging
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from src.data.make_dataset import DatasetRecipes
from src.models.ViT import ViT

from pathlib import Path

from tqdm import tqdm

import skimage.measure


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(data_path, n_epochs=20, batch_size=16, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_seed(
        seed=seed
    )  # To make sure, because the same seed will be used for test set in another file

    train_path = data_path / "train"
    validation_path = data_path / "validation"

    train_dataset = DatasetRecipes(train_path)
    validation_dataset = DatasetRecipes(validation_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    img_size = train_dataset[0][0].shape[-2:]

    model = ViT(
        img_dims=img_size,
        channels=3,
    )
    model.to(device)

    for epoch in tqdm(range(n_epochs)):
        for data in train_loader:
            img, dt = data

            img = img.to(device)

            out = model(img)

            print(out.shape)
            return


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    options = {
        "data_path": Path(r"data/processed"),
        "batch_size": 16,
        "n_epochs": 20,
        "seed": 0,
    }

    main(**options)
