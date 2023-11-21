import torch
import torch.nn as nn

from src.models import train_ViT_Txt_Transf


def main():
    # Load the model I want.

    # Load its parameters/state_dict from the respective .pt file in models/

    # Load a preprocessed file with text embeddings. Or Create this in real time if possible.
    # This file is created by having a text dataset and passing each point through the text part of the model.
    # Get the embeddings and append to a tensor. Probably a large one in the end.
    # If I have generated the file, I do not need the tokenizer. Otherwise I need it for this part

    # Pass the image through the corresponding model part, get embedding

    # Compare against all other text embeddings, get topk(1) or argmax

    # Return the text at the argmax position

    pass


if __name__ == "__main__":
    main()
