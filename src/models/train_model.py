import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.make_dataset import DatasetRecipes
from src.models import ViT


def main(data_path, n_epochs=20, batch_size=16, seed=0):
    dataset = DatasetRecipes(data_path)

    torch.manual_seed(seed)
    train_set, val_set = # Split with seed. Use the same seed for test set when using inference

    train_loader = DataLoader(train_set,batch_size, shuffle=True)
    val_loader = DataLoader(val_set,batch_size, shuffle=False)

    logger.info(len(dataset))


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
