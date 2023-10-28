import logging
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.make_dataset import DatasetRecipes
from src.models.ViT import ViT


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(data_path, n_epochs=20, batch_size=16, seed=0, splits_sizes=(0.8, 0.1, 0.1)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_seed(
        seed=seed
    )  # To make sure, because the same seed will be used for test set in another file

    dataset = DatasetRecipes(data_path)

    train_size, val_size, test_size = splits_sizes

    train_dataset, val_dataset, _ = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    options = {
        "data_path": r"data/raw/archive",
        "batch_size": 16,
        "n_epochs": 20,
        "seed": 0,
    }

    main(**options)
