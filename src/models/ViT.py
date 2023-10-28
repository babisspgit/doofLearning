import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super(Attention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim * (-0.5)

        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()

        keys = self.k_projection(x)
        queries = self.q_projection(x)
        values = self.v_projection(x)

        keys = rearrange(
            keys, "b seq (h d) -> (b h) seq d", h=self.num_heads, d=self.head_dim
        )
        values = rearrange(
            values, "b seq (h d) -> (b h) seq d", h=self.num_heads, d=self.head_dim
        )
        queries = rearrange(
            queries, "b seq (h d) -> (b h) seq d", h=self.num_heads, d=self.head_dim
        )

        attention_logits = torch.matmul(keys, values.transpose(1, 2))
        attention_logits *= self.scale
        attention = torch.nn.functional.softmax(attention_logits, dim=-1)

        out = torch.matmul(attention, values)

        # Rearrange output from all heads
        out = rearrange(
            out, "(b h) seq d->b seq (h d)", h=self.num_heads, d=self.head_dim
        )

        # Remember to add aseertations

        return self.o_projection(out)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_hidden_dims=None, dropout=0.0):
        super(EncoderBlock, self).__init__()

        # The transformer's encoder block has residual connections
        # Define layers one by one

        self.attention = Attention(
            num_heads, embed_dim
        )  # Can use nn.MultiheadAttention inst of ours

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Arbitrary selection of linear hidden layer dims. Can be passed in constructor
        self.fc_hidden_dims = (
            4 * embed_dim if fc_hidden_dims is None else fc_hidden_dims
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fc_hidden_dims),
            nn.GELU(),
            nn.LayerNorm(fc_hidden_dims),
            nn.Linear(fc_hidden_dims, embed_dim),
        )

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.layer_norm1(x + attention_output)
        x = self.dropout(x)

        fc_out = self.fc(x)

        x = self.layer_norm2(fc_out + x)
        x = self.dropout(x)

        return x


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
        for i in num_layers:
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
        positions = self.positional_embedding.to(x.device, dtype=x.dtype)
        x = positions + x

        x = self.dropout(x)
        x = self.transformer(x)

        # Find a way to pool the last layer and project on embedding dims
        # Maybe add a linear layer or just let it be, since the
        # output is already in embed_dim dims
        pass
