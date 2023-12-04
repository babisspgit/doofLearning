import torch
import torch.nn as nn
from src.models.Transformers_ import ViT, TextTransformer, bert_model, pretrained_vit
from src.models.CNN_models import VGG, VGGImageEncoder


class PretrainedViT_Multiple(nn.Module):
    def __init__(self, vit_config_name: str, txt_options_1: dict, txt_options_2: dict):
        super().__init__()

        self.img_model = pretrained_vit(vit_config_name)

        self.text_model_1 = TextTransformer(**txt_options_1)  # or bert
        self.text_model_2 = TextTransformer(**txt_options_2)  # or bert

    def forward(self, img_tensor, tok_text_tensor_1, tok_text_tensor_2):
        img_embeddings = self.img_model(img_tensor)
        text_embeddings_1 = self.text_model_1(tok_text_tensor_1)
        text_embeddings_2 = self.text_model_2(tok_text_tensor_2)

        text_embeddings = torch.cat((text_embeddings_1, text_embeddings_2), dim=1)

        assert (
            img_embeddings.shape[1] == text_embeddings.shape[1]
        ), "Image and concatenated text embeddings have different shapes"

        img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )

        # Create the cosine similarities as a matrix
        logits_per_text = torch.matmul(text_embeddings, img_embeddings.t())
        logits_per_image = logits_per_text.t()

        return logits_per_text, logits_per_image, text_embeddings, img_embeddings


class PretrainedViT(nn.Module):
    def __init__(self, vit_config_name: str, txt_options: dict):
        super().__init__()

        self.img_model = pretrained_vit(vit_config_name)
        self.text_model = TextTransformer(**txt_options)  # or bert

    def forward(self, img_tensor, tok_text_tensor):
        img_embeddings = self.img_model(img_tensor)
        text_embeddings = self.text_model(tok_text_tensor)

        img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )

        # Create the cosine similarities as a matrix
        logits_per_text = torch.matmul(text_embeddings, img_embeddings.t())
        logits_per_image = logits_per_text.t()

        return logits_per_text, logits_per_image, text_embeddings, img_embeddings


## Vit + TextTransformer
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

        return logits_per_text, logits_per_image, text_embeddings, img_embeddings


## Vit + bert
class Transformers_Bert(nn.Module):
    def __init__(self, img_options_dict: dict, logit_scale_init_value=None) -> None:
        super().__init__()

        output_txt_dim = img_options_dict["embed_dim"]

        self.img_model = ViT(**img_options_dict)
        self.text_model = bert_model(output_dim=output_txt_dim)

        if logit_scale_init_value:
            self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))
        else:
            self.logit_scale = torch.tensor(0.0)

    def forward(self, img_tensor, tok_text_tensor, attention_mask=None):
        img_embeddings = self.img_model(img_tensor)
        text_embeddings = self.text_model(
            tok_text_tensor, attention_mask=attention_mask
        )

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

        return text_embeddings, img_embeddings, logits_per_text, logits_per_image


class Transformers_Bert2(nn.Module):
    def __init__(
        self,
        img_options_dict: dict,
        text_options_dict: dict,
        logit_scale_init_value: float = None,
    ) -> None:
        super().__init__()

        self.img_model = ViT(**img_options_dict)

        self.text_model = bert_model()  # (**text_options_dict)

        # return img_out, text_out

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


## VGG + TextTransformer
class VGG_SingleTextModel(nn.Module):  ## works!
    def __init__(self, img_options_dict: dict, text_options_dict: dict) -> None:
        super().__init__()
        self.embedding_size = 512
        self.img_model = VGG(self.embedding_size)
        self.text_model = TextTransformer(**text_options_dict)

    def forward(self, img_tensor, tok_text_tensor):
        img_out = self.img_model(img_tensor)
        text_out = self.text_model(tok_text_tensor)

        return img_out, text_out


# pretrained VGG + TextTransformer
class VGGpre_SingleTextModel(nn.Module):  ## works!
    def __init__(self, img_options_dict: dict, text_options_dict: dict) -> None:
        super().__init__()
        self.embedding_size = 512
        # Define the output dimension for your image embeddings
        self.img_model = VGGImageEncoder(self.embedding_size)
        # i want to print the structure of the model
        print(self.img_model)
        self.text_model = TextTransformer(**text_options_dict)

    def forward(self, img_tensor, tok_text_tensor):
        img_out = self.img_model(img_tensor)
        text_out = self.text_model(tok_text_tensor)

        return img_out, text_out


## VGG + bert
class VGG_Bert(nn.Module):
    def __init__(self, img_options_dict: dict, text_options_dict: dict) -> None:
        super().__init__()

        self.img_model = VGG(**img_options_dict)
        self.text_model = bert_model(text_options_dict)

    def forward(self, img_tensor, tok_text_tensor):
        img_out = self.img_model(img_tensor)
        text_out = self.text_model(tok_text_tensor)

        return img_out, text_out


# pretrained
class VGGpre_Bert(nn.Module):
    def __init__(self, img_options_dict: dict, text_options_dict: dict) -> None:
        super().__init__()

        outputimg_dim = 128
        outputtxt_dim = 128

        self.img_model = VGGImageEncoder(output_dim=outputimg_dim)
        self.text_model = bert_model(output_dim=outputtxt_dim)

    def forward(self, img_tensor, tok_text_tensor):
        img_out = self.img_model(img_tensor)
        text_out = self.text_model(tok_text_tensor)

        return img_out, text_out
