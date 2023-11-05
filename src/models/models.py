import torch
import torch.nn as nn
from src.models.Transformers_ import ViT, TextTransformer, bert_model


class TransformersSingleTextModel(nn.Module):
    def __init__(
        self,
        img_options_dict: dict,
        text_options_dict: dict,
        logit_scale_init_value: float = None,
    ) -> None:
        super().__init__()

        self.img_model = ViT(**img_options_dict)
        self.text_model = TextTransformer(**text_options_dict)

        # Magic or learnable number? Found in clip
        if logit_scale_init_value:
            self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))
        else:
            self.logit_scale = torch.tensor(0.0)
        # self.text_model = bert_model

    def forward(self, img_tensor, tok_text_tensor):
        img_embeddings = self.img_model(img_tensor)
        text_embeddings = self.text_model(tok_text_tensor)

        img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )

        # Create the cosine similarities as a matrix
        logit_scale = self.logit_scale.exp()
        logits_per_text = (
            torch.matmul(text_embeddings, img_embeddings.t()) * logit_scale
        )
        logits_per_image = logits_per_text.t()

        return logits_per_text, logits_per_image


class CLIP_(nn.Module):
    def __init__(self, pretrained: str = "") -> None:
        # Pretrained:
        # "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(pretrained)

    def forward(self, img, text):
        return self.clip_model(img, text)
