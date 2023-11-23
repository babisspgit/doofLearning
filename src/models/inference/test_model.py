import logging


import torch
import torch.nn as nn

from src.models import train_ViT_Txt_Transf

from pathlib import Path

import pandas as pd


def main():
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

    assert text_embeddings.shape[0] == len(
        recipes_df
    ), "Number of text embeddings is not the same as the number of recipes"

    return

    # Load the model I want. Careful to use the same model that generated the embeddings
    model_options = {
        # Model options
    }
    # model = # Model

    # Load its parameters/state_dict from the respective .pt file in models/
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Load a preprocessed file with text embeddings. Or Create this in real time if possible.
    # This file is created by having a text dataset and passing each point through the text part of the model.
    # Get the embeddings and append to a tensor. Probably a large one in the end.
    # If I have generated the file, I do not need the tokenizer. Otherwise I need it for this part

    # Pass the image through the corresponding model part, get embedding

    # Compare against all other text embeddings, get topk(1) or argmax

    # Return the text at the argmax position

    pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    main()
