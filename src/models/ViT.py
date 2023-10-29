import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from src.models.TransformerBlock import EncoderBlock


class ViT(nn.Module):
    def __init__(
        self,
        img_dims,
        channels,
        patch_sizes,
        embed_dim,
        num_heads,
        num_layers,
        fc_hidden_dims=None,
        dropout=0.0,
    ):
        super(ViT, self).__init__()

        height, width = img_dims
        patch_h_size, patch_w_size = patch_sizes

        assert (
            height % patch_h_size == 0 and width % patch_w_size == 0
        ), "Image dims must be divisible by patch sizes"

        num_patches = (height // patch_h_size) * (width // patch_w_size)
        patch_dim = channels * patch_h_size * patch_w_size

        # From input image to flattened patches
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (nph ph) (npw pw)-> b (nph npw) (c ph pw)",
                ph=patch_h_size,
                pw=patch_w_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Use learnable positional embeddings for ease
        self.positional_embeddings = nn.Parameter(torch.rand(1, num_patches, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Encoder layers
        enc_layers = []
        for i in range(num_layers):
            enc_layers.append(
                EncoderBlock(embed_dim, num_heads, fc_hidden_dims, dropout)
            )

        self.transformer = nn.Sequential(*enc_layers)

    def forward(self, x):
        # x is an img tensor
        # batch_size, 3, h, w
        # Tokenize and flatten

        x = self.to_patch_embedding(x)

        # add positional embedding to each patch
        positions = self.positional_embeddings.to(x.device, dtype=x.dtype)
        x = positions + x

        x = self.dropout(x)
        x = self.transformer(x)

        # Find a way to pool the last layer and project on embedding dims
        # Maybe add a linear layer or just let it be, since the
        # output is already in embed_dim dims
        return x
