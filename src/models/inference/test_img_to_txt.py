import logging


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from src.data.make_dataset import DatasetRecipes
from src.utils.vocab_build import get_vocab

from pathlib import Path

import pandas as pd


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on: {device}")

    # Paths
    recipes_df_path = Path("models/inference/recipes.csv")

    # Load the recipes dataframe to get the text answer
    logger.info("Loading csv")
    recipes_df = pd.read_csv(recipes_df_path)

    # Get the embeddings
    logger.info("Loading text embeddings")
    text_embeddings_path = Path(
        "models/inference/embeddings/ViT_Text/embeddings_Title__text.pt"
    )
    text_embeddings = torch.load(text_embeddings_path)

    # Normalize again in case they were not normalized when saving!
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

    assert text_embeddings.shape[0] == len(
        recipes_df
    ), "Number of text embeddings is not the same as the number of recipes"

    saved_model_path = Path("models/ViT_Text_Transf_full.pt")
    model = torch.load(saved_model_path)
    model.eval()
    model.to(device)

    # Get an image. For the test set, this is better done bu a dataloader object
    data_path = Path("data/processed/test")

    columns = ["Title"]
    test_set = DatasetRecipes(
        data_path=data_path, columns=columns, transformations=None
    )

    batch_size = 20
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    img, real_text = next(iter(test_loader))

    img = img.to(device)

    img_embeddings = model.img_model(img)

    # Normalize mebeddings
    img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()

    logits_per_img = torch.matmul(img_embeddings, text_embeddings.t()) * logit_scale

    # Get the max index
    idx = torch.argmax(logits_per_img, dim=1)  # or topk, but need to change code below

    for i, rec_idx in enumerate(idx):
        rec_idx = rec_idx.item()
        print(f"Target: {real_text[i]}")
        print(f"Pred: {recipes_df.iloc[rec_idx].Title}")
        print("*" * 25)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    main()
