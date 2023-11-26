import logging


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from src.data.make_dataset import DatasetRecipes
from src.utils.vocab_build import get_vocab

from pathlib import Path

import pandas as pd


from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import cv2
import einops
from einops import rearrange





def show_mask_on_image(img, mask):
    if img.max() > 1:
        img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def visualize_attention_maps(
    model, image, head_fusion, discard_ratio, file_name="attention_maps.pdf", image_title='title'
):
    """
    https://jacobgil.github.io/deeplearning/vision-transformer-explainability
    """

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Pick n_images images from one dataloader, prob validation

    img = image.to(device)#  dataset[1][0].to(device)  # plus 1 to have the cat

    # Pass the image throught the model, getting the attention output of the last encoder layer
    # Image to patches
    tokens = model.img_model.to_patch_embedding(img.unsqueeze(0))

    # # Add cls token
    # tokens = torch.cat([model.cls_token, tokens], dim=1)

    # Add positional embeddings
    positions = model.img_model.positional_embeddings.to(img.device, dtype=img.dtype)

    x = tokens + positions

    # Pass each input through the transformer (attention) blocks
    attentions = []
    for i, block in enumerate(model.img_model.transformer):
        x = block(x)

        keys = block.attention.k_projection(x)
        queries = block.attention.q_projection(x)
        values = block.attention.v_projection(x)

        keys = rearrange(
            keys, "b seq (h d) -> (b h) seq d", h=block.attention.num_heads, d=block.attention.head_dim
        )
        values = rearrange(
            values, "b seq (h d) -> (b h) seq d", h=block.attention.num_heads, d=block.attention.head_dim
        )
        queries = rearrange(
            queries, "b seq (h d) -> (b h) seq d", h=block.attention.num_heads, d=block.attention.head_dim
        )

        attention_logits = torch.matmul(keys, values.transpose(1, 2))
        attention_logits *= block.attention.scale
        attention = torch.nn.functional.softmax(attention_logits, dim=-1)




        attentions.append(attention.cpu())


    # Attentions from array to tensor
    n_blocks = len(attentions)
    attentions_t = torch.zeros_like(attentions[-1])
    attentions_t = einops.repeat(attentions_t, "h t e -> a h t e", a=n_blocks)

    for i in range(len(attentions)):
        attentions_t[i] = attentions[i]

    # print(attentions_t.shape)

    # Rollout algorithm
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for i, attention in enumerate(attentions_t):
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=0)  # axis=1
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=0)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=0)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)

            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))

            a = (attention_heads_fused + 1.0 * I) / 2

            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches

    print(result.shape)
    # mask = result[0, 1:]
    mask = result[:]
    # print(mask.shape, mask.max(), img.max())


    mask = mask.reshape(1, -1)

    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()

    # print(f"After mask shape: {mask.shape}")

    mask = mask / np.max(mask)

    img = img.permute(1, 2, 0).cpu().detach().numpy()

    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)

    fig = plt.figure()

    plt.suptitle(image_title)

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.tight_layout()
    plt.axis('Off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, ::-1])
    plt.tight_layout()
    plt.axis('Off')

    plt.savefig(file_name)
    plt.show()

    return




def get_image(root_path, image_name, extension=".jpg"):
    root_path = Path(root_path)
    image_name += extension
    try:
        img_path = root_path / "train" / "images" / image_name
        img = Image.open(img_path)
        return img
    except FileNotFoundError:
        try:
            img_path = root_path / "validation" / "images" / image_name
            img = Image.open(img_path)
            return img
        except FileNotFoundError:
            print("Not found")
            return None


def main():

    # Predictions path
    predictions_saved_path = Path('data/img2txt')
    predictions_saved_path.mkdir(exist_ok=True, parents=True)

    # Images root
    images_paths = Path("data/processed")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on: {device}")

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

    # Normalize again in case they were not normalized when saving!
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

    assert text_embeddings.shape[0] == len(
        recipes_df
    ), "Number of text embeddings is not the same as the number of recipes"

    saved_model_path = Path("models/ViT_Text_Transf_full.pt")
    model = torch.load(saved_model_path)
    model.eval()
    model.to(device)

    # Get an image. For the test set, this is better done bu a dataloader object
    data_path = Path("data/processed/test")

    columns = ["Title"]
    test_set = DatasetRecipes(
        data_path=data_path, columns=columns, transformations=None
    )





    batch_size = 20
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    img, real_text = next(iter(test_loader))

    img = img.to(device)

    img_embeddings = model.img_model(img)

    # Normalize mebeddings
    img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()

    logits_per_img = torch.matmul(img_embeddings, text_embeddings.t()) * logit_scale

    # Get the max index
    idx = torch.argmax(logits_per_img, dim=1)  # or topk, but need to change code below

    # for i, rec_idx in enumerate(idx):
    #     rec_idx = rec_idx.item()
    #     print(f"Target: {real_text[i]}")
    #     for col in columns:
    #         print(f"Pred: {recipes_df.iloc[rec_idx][col]}")
    #     print("*" * 25)


    for i, rec_idx in enumerate(idx):
        rec_idx = rec_idx.item()
        
        pred_image_name = recipes_df.iloc[rec_idx].Image_Name
        print(pred_image_name)
        # image = get_image(images_paths, pred_image_name)
        image = img[i].permute((1,2,0)).detach().cpu().numpy()

        img_title = f"Prediction:\n{recipes_df.iloc[rec_idx]['Title']}"
        

        # # image = image.convert('RGB')
        # plt.imshow(image)
        # plt.title(f"Prediction:\n{recipes_df.iloc[rec_idx]['Title']}")
        # plt.axis('Off')

        # plt.tight_layout()
        # save_name = predictions_saved_path / f"pred_for_{pred_image_name}_real_{real_text[i]}.jpg"
        # plt.savefig(save_name)
        # plt.show()

        save_name = predictions_saved_path / f"pred_for_{pred_image_name}_real_{real_text[i]}_attention.jpg"
        visualize_attention_maps(
        model, img[i], 'max', discard_ratio=0.1, file_name=save_name, image_title=img_title
    )

    print("*" * 25)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    main()
