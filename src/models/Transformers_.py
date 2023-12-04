import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
from transformers import BertModel, ViTModel, ViTForImageClassification


import math


class Attention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super(Attention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** (-0.5)

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

        attention_logits = torch.matmul(keys, queries.transpose(1, 2))
        attention_logits *= self.scale
        attention = torch.nn.functional.softmax(attention_logits, dim=-1)

        out = torch.matmul(attention, values)

        # Rearrange output from all heads
        out = rearrange(
            out, "(b h) seq d->b seq (h d)", h=self.num_heads, d=self.head_dim
        )

        # Remember to add assertations

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
            nn.Linear(embed_dim, self.fc_hidden_dims),
            nn.GELU(),
            nn.LayerNorm(self.fc_hidden_dims),
            nn.Linear(self.fc_hidden_dims, embed_dim),
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
        # projection_dims,
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
        for _ in range(num_layers):
            enc_layers.append(
                EncoderBlock(embed_dim, num_heads, fc_hidden_dims, dropout)
            )

        self.transformer = nn.Sequential(*enc_layers)

        # self.projection_layer = nn.Linear(
        #     embed_dim, projection_dims
        # )  # Equivalent to W_i from CLIP

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
        # if self.pool == "max":
        #     x = x.max(dim=1)[0]
        # elif self.pool == "mean":
        #     x = x.mean(dim=1)

        print(x.shape)

        x = x.max(dim=1)[0]

        print(x.shape)
        return x
        # return self.projection_layer(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0.0, max_seq_len).unsqueeze(1)

        ####
        # From https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        ####

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x + self.pe[:, :seq_length]
        # return self.dropout(x)


class TextTransformer(nn.Module):
    def __init__(
        self,
        num_heads,
        num_blocks,
        embed_dims,
        # projection_dims,
        vocab_size,
        max_seq_len,
        dropout=0.0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            embedding_dim=embed_dims, num_embeddings=vocab_size
        )

        self.positional_encoding = PositionalEncoding(embed_dims, max_seq_len)

        encoder_blocks = []
        for _ in range(num_blocks):
            encoder_blocks.append(
                EncoderBlock(embed_dim=embed_dims, num_heads=num_heads)
            )

        self.text_transformer_blocks = nn.Sequential(*encoder_blocks)

        # self.projection_layer = nn.Linear(
        #     embed_dims, projection_dims
        # )  # Equivalent to W_t from CLIP

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Check if I need to pad the text!
        tokens = self.embedding(x)
        x = self.positional_encoding(tokens)
        x = self.dropout(x)
        # text inst of x since I do not yet know if I will need
        # to use the length from the dataloader, collate_fn

        x = self.text_transformer_blocks(x)

        # Probably need to say sth about max seq
        # and also pooling the last attention output
        x = x.max(dim=1)[0]
        return x
        # return self.projection_layer(x)


# def bert_model():
#    model = BertModel.from_pretrained('bert-base-uncased')

# `
#   return model


from transformers import BertModel
import torch.nn as nn


class bert_model(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Load the pretrained BERT model
        # self.embedding_dim =
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Projection layer to map BERT's output to the desired embedding dimension
        self.projection_layer = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask=None):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Take the outputs from the last hidden state
        pooling_output = outputs.pooler_output

        # Apply the projection layer to every token output
        embeddings = self.projection_layer(pooling_output)

        return embeddings


class pretrained_vit(nn.Module):
    def __init__(self, vit_config_name: str = None):
        super().__init__()

        model_name_or_path = vit_config_name
        if not vit_config_name:
            model_name_or_path = "google/vit-base-patch16-224-in21k"

        # self.model = ViTModel.from_pretrained(model_name_or_path)
        self.model = ViTModel(model_name_or_path)

        # self.project_layer = nn.Linear(output_of_last_hidden_stat ,embed_dim)

    def forward(self, x):
        x = self.model(x)
        a = x.pooler_output
        return a
