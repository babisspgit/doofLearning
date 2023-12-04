import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from pathlib import Path
from tqdm import tqdm

# from src.models.models import TransformersSingleTextModel
# from src.utils.vocab_build import CustomTokenizer, get_vocab

# from src.data.make_dataset import DatasetRecipes


from src.data.make_dataset import (
    DatasetRecipesSep,
    DatasetRecipes,
    DatasetRecipesSepTriplet,
)
from src.models.models import PretrainedViT
from transformers import ViTImageProcessor, ViTConfig
from src.utils.vocab_build import get_vocab, CustomTokenizer


import hydra
from omegaconf import OmegaConf


@hydra.main(
    config_path="../config/ViT_TxtTransfConfig",
    config_name="default_config.yaml",
    version_base=None,
)
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack hparams
    # seed = config.seed

    # lr = config.lr
    batch_size = config.batch_size
    # n_epochs = config.n_epochs
    # save_per_n_epochs = config.save_per_n_epochs

    # data path(s)
    data_path = Path(config.data_path)

    # Unpack experiment specific params
    hparams = config["_group_"]  # wtf is this __group__ ?

    embed_dim = hparams.embed_dim

    # ViT
    image_dims = hparams.vit.image_dims
    # patch_dims = hparams.vit.patch_dims
    # num_heads_vit = hparams.vit.num_heads
    # num_blocks_vit = hparams.vit.num_blocks

    # # Text Transformer
    # num_heads_text = hparams.text_transf.num_heads
    # num_blocks_text = hparams.text_transf.num_blocks

    # data paths
    train_path = data_path / "train"
    validation_path = data_path / "validation"

    # datasets
    train_transform = transforms.Compose(
        [
            transforms.Resize([image_dims, image_dims]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    columns = ["Title"]

    train_dataset_vocab = DatasetRecipes(
        train_path, columns=columns, transformations=train_transform
    )

    train_dataset = DatasetRecipesSepTriplet(train_path)
    val_dataset = DatasetRecipesSepTriplet(validation_path)

    # Vocab and tokenizers
    tokenizer = CustomTokenizer()

    cust_name = "_".join(columns)
    vocab_path = Path(f"models/simple_vocab_{cust_name}")

    vocab_, MAX_SEQ_LEN = get_vocab(
        train_dataset_vocab, tokenizer=tokenizer.tokenize, vocab_save_path=vocab_path
    )

    VOCAB_SIZE = len(vocab_)

    saved_model_path = Path("models/Pretrained_ViT_Text_Transf_Triplet_full.pt")
    model = torch.load(saved_model_path)
    # model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    model.to(device)

    config = ViTConfig(
        image_size=image_dims, hidden_size=embed_dim, num_attention_heads=8
    )
    # config = "google/vit-base-patch16-224-in21k"
    img_processor = ViTImageProcessor(config, size=image_dims)

    def process_text(raw_text):
        # Pipeline
        text_pipeline = lambda x: [vocab_[token] for token in tokenizer.tokenize(x)]

        # Process
        if len(raw_text) > MAX_SEQ_LEN - 2:
            raw_text = raw_text[: MAX_SEQ_LEN - 2]

        processed_text = text_pipeline(raw_text)

        # Insert bos token and eos token

        processed_text = [vocab_["<bos>"]] + processed_text + [vocab_["<eos>"]]

        # pad
        # NOTE: I am padding all the way from the right ONLY, torch uses a padding method from both sides
        pad = [vocab_["<pad>"]] * (MAX_SEQ_LEN - len(processed_text))

        processed_text = processed_text + pad

        processed_text = torch.tensor(processed_text, dtype=torch.int64)

        return processed_text

    def collate_batch(batch):
        anchor_img_list, anchor_text_list = [], []
        neg_img_list, neg_text_list = [], []
        for a in batch:  # only title in this case hardcoded
            # Protect against larger sequencies
            # -2 to account for BOS and EOS tokens
            anch_dict = a[0]  # for 1 image anchor, pos and neg text
            negs_dict = a[1]  # for 1 text anchor, pos and neg images

            # Unpacking
            anchor_image = anch_dict["img"]
            anchor_text = anch_dict["title"]

            neg_img = negs_dict["img"]
            neg_text = negs_dict["title"]

            # Process the texts
            processed_anchor_text = process_text(anchor_text)
            processed_neg_text = process_text(neg_text)

            # Build the batch lists
            anchor_text_list.append(processed_anchor_text.unsqueeze(0))
            neg_text_list.append(processed_neg_text.unsqueeze(0))

            # img_list.append(img_processor(img, return_tensors="pt")["pixel_values"])

            anchor_img_list.append(
                img_processor(anchor_image, return_tensors="pt")["pixel_values"]
            )
            neg_img_list.append(
                img_processor(neg_img, return_tensors="pt")["pixel_values"]
            )

        # Return dictionaries, like the dataset class in __getitem__

        anchors_dict = {
            "img": torch.cat(anchor_img_list, axis=0).to(device),
            "text": torch.cat(anchor_text_list, axis=0).to(device),
        }

        negs_dict = {
            "img": torch.cat(neg_img_list, axis=0).to(device),
            "text": torch.cat(neg_text_list, axis=0).to(device),
        }

        return anchors_dict, negs_dict

    ## Careful not to shuffle!
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    ### Define a save name for the embeddings. Should be descriptive
    text_save_path = Path("models/inference/embeddings") / "pretrained_ViT_Text_Triplet"
    text_save_path.mkdir(exist_ok=True, parents=True)

    img_save_path = text_save_path / f"embeddings_{cust_name}__img.pt"
    text_save_path = text_save_path / f"embeddings_{cust_name}__text.pt"

    # Try a small part of the dataset
    text_embeddings_stacked = torch.zeros((1, embed_dim)).to(device)
    img_embeddings_stacked = torch.zeros((1, embed_dim)).to(device)

    pbar = tqdm(train_loader)
    for batch in pbar:
        with torch.no_grad():
            anchors_dict, _ = batch

            anchor_img = anchors_dict["img"]
            anchor_text = anchors_dict["text"]

            img_embeddings = model.img_model(anchor_img)
            text_embeddings = model.text_model(anchor_text)

            # Normalize the embeddings
            img_embeddings = img_embeddings / img_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )
            text_embeddings = text_embeddings / text_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )

            text_embeddings_stacked = torch.vstack(
                [text_embeddings_stacked, text_embeddings]
            )

            img_embeddings_stacked = torch.vstack(
                [img_embeddings_stacked, img_embeddings]
            )

    pbar = tqdm(val_loader)
    for batch in pbar:
        with torch.no_grad():
            anchors_dict, _ = batch

            anchor_img = anchors_dict["img"]
            anchor_text = anchors_dict["text"]

            img_embeddings = model.img_model(anchor_img)
            text_embeddings = model.text_model(anchor_text)

            img_embeddings = img_embeddings / img_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )
            text_embeddings = text_embeddings / text_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )

            text_embeddings_stacked = torch.vstack(
                [text_embeddings_stacked, text_embeddings]
            )

            img_embeddings_stacked = torch.vstack(
                [img_embeddings_stacked, img_embeddings]
            )

    torch.save(text_embeddings_stacked[1:], text_save_path)
    torch.save(img_embeddings_stacked[1:], img_save_path)


if __name__ == "__main__":
    main()
