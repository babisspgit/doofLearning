import torch
import torch.nn as nn

from src.models.TransformerBlock import EncoderBlock


class TextTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Encode the text, use nn.Embedding for now
        pass
