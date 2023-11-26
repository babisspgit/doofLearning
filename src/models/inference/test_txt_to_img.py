from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader

from PIL import Image
from src.data.make_dataset import DatasetRecipes
from src.utils.vocab_build import CustomTokenizer, get_vocab

import matplotlib.pyplot as plt

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
    predictions_saved_path = Path('data/txt2img')
    predictions_saved_path.mkdir(exist_ok=True, parents=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    # Load recipes dataframe, where the corresponding image is stored
    recipes_df_path = Path("models/inference/recipes.csv")

    # Load the recipes dataframe to get the text answer
    print("Loading csv")
    recipes_df = pd.read_csv(recipes_df_path)

    # Get the embeddings
    print("Loading text embeddings")
    img_embeddings_path = Path(
        "models/inference/embeddings/ViT_Text/embeddings_Title__img.pt"
    )
    img_embeddings = torch.load(img_embeddings_path)

    # Normalize again in case they were not normalized when saving!
    img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)

    assert img_embeddings.shape[0] == len(
        recipes_df
    ), "Number of text embeddings is not the same as the number of recipes"

    # Get the model
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

    # Here I need the collate fn, or at least
    # the text processing pipeline

    tokenizer = CustomTokenizer()

    cust_name = "_".join(columns)
    vocab_path = Path(f"models/simple_vocab_{cust_name}")

    vocab_, MAX_SEQ_LEN = get_vocab(
        test_set, tokenizer=tokenizer.tokenize, vocab_save_path=vocab_path
    )

    VOCAB_SIZE = len(vocab_)

    text_pipeline = lambda x: [vocab_[token] for token in tokenizer.tokenize(x)]


    def collate_batch(batch):
        img_list, text_list, real_text_list = [], [], []
        for img, _text in batch:
            real_text_list.append(_text)
            if len(_text) > MAX_SEQ_LEN - 2:
                _text = _text[: MAX_SEQ_LEN - 2]

            processed_text = text_pipeline(_text)

            processed_text = [vocab_["<bos>"]] + processed_text + [vocab_["<eos>"]]

            pad = [vocab_["<pad>"]] * (MAX_SEQ_LEN - len(processed_text))

            processed_text = processed_text + pad

            processed_text = torch.tensor(processed_text, dtype=torch.int64)

            text_list.append(processed_text.unsqueeze(0))


            img_list.append(img.unsqueeze(0))

        return (
            torch.cat(img_list, axis=0).to(device),
            torch.cat(text_list, axis=0).to(device),
            real_text_list,
        )

    # Images root
    images_paths = Path("data/processed")

    # Get a text data point

    batch_size = 20
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    _, text, real_text = next(iter(test_loader))

    text = text.to(device)
    txt_embeddings = model.text_model(text)

    # Normalize mebeddings
    txt_embeddings = txt_embeddings / txt_embeddings.norm(p=2, dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()

    logits_per_text = torch.matmul(txt_embeddings, img_embeddings.t()) * logit_scale

    # Get the max index
    idx = torch.argmax(logits_per_text, dim=1)  # or topk, but need to change code below

    for i, rec_idx in enumerate(idx):
        rec_idx = rec_idx.item()
        # print(f"Target: {real_text[i]}")
        pred_image_name = recipes_df.iloc[rec_idx].Image_Name
        print(pred_image_name)
        image = get_image(images_paths, pred_image_name)
        if image:
            # save_name = predictions_saved_path / f"pred_for_{real_text[i]}.jpg"
            # image.save(save_name)
            image = image.convert('RGB')
            plt.imshow(image)
            plt.title(f"User input:\n{real_text[i]}")
            plt.axis('Off')

            plt.tight_layout()
            save_name = predictions_saved_path / f"pred_for_{real_text[i]}_real_{pred_image_name}.jpg"
            plt.savefig(save_name)
            plt.show()
        print("*" * 25)

if __name__ == "__main__":
    main()
