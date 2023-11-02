import torch.nn as nn
from src.models.Transformers_ import ViT, TextTransformer,bert_model

class TransformersSingleTextModel(nn.Module):
    def __init__(self, img_options_dict: dict, text_options_dict: dict) -> None:
        super().__init__()

        self.img_model = ViT(**img_options_dict)
        self.text_model = TextTransformer(**text_options_dict)
        #self.text_model = bert_model 
        
    def forward(self, img_tensor, tok_text_tensor):
        img_out = self.img_model(img_tensor)
        text_out = self.text_model(tok_text_tensor)

        return img_out, text_out
