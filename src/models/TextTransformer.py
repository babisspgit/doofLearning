import torch
import torch.nn as nn

from src.models.TransformerBlock import EncoderBlock


class TextTransformer(nn.Module):
    def __init__(self, num_heads, num_blocks, embed_dims, projection_dims, seq_len ):
        super().__init__()
        
        self.embedding = nn.Embedding(seq_len, embed_dims)


        encoder_blocks = []
        for _ in range(num_blocks):
            encoder_blocks.append(
                EncoderBlock(embed_dim=embed_dims,
                             num_heads=num_heads)
            )

        self.text_transformer = nn.Sequential(*encoder_blocks)

        self.projection_layer = nn.Linear(embed_dims, projection_dims)


    def forward(self, x):

        # Check if I need to pad the text!

        # text inst of x since I do not yet know if I will need
        # to use the length from the dataloader, collate_fn
        x = self.embedding(x)

        x = self.text_transformer(x)

        # Probably need to say sth about max seq
        # and also pooling the last attention output
        x = x.max(dim=1)[0]
        return self.projection_layer(x)
