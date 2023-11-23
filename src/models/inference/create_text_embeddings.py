import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from pathlib import Path
from tqdm import tqdm

from src.models.models import TransformersSingleTextModel
from src.utils.vocab_build import CustomTokenizer, get_vocab

from src.data.make_dataset import DatasetRecipes

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
    seed = config.seed

    lr = config.lr
    batch_size = config.batch_size
    n_epochs = config.n_epochs
    save_per_n_epochs = config.save_per_n_epochs

    # data path(s)
    data_path = Path(config.data_path)

    # Unpack experiment specific params
    hparams = config["_group_"]  # wtf is this __group__ ?

    embed_dim = hparams.embed_dim

    # ViT
    image_dims = hparams.vit.image_dims
    patch_dims = hparams.vit.patch_dims
    num_heads_vit = hparams.vit.num_heads
    num_blocks_vit = hparams.vit.num_blocks

    # Text Transformer
    num_heads_text = hparams.text_transf.num_heads
    num_blocks_text = hparams.text_transf.num_blocks

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

    train_dataset = DatasetRecipes(
        train_path, columns=columns, transformations=train_transform
    )
    val_dataset = DatasetRecipes(
        validation_path, columns=columns, transformations=train_transform
    )

    # Vocab and tokenizers
    tokenizer = CustomTokenizer()

    cust_name = "_".join(columns)
    vocab_path = Path(f"models/simple_vocab_{cust_name}")

    vocab_, MAX_SEQ_LEN = get_vocab(
        train_dataset, tokenizer=tokenizer.tokenize, vocab_save_path=vocab_path
    )

    VOCAB_SIZE = len(vocab_)

    vit_options = {
        "img_dims": (image_dims, image_dims),
        "channels": 3,
        "patch_sizes": (patch_dims, patch_dims),
        "embed_dim": embed_dim,
        "num_heads": num_heads_vit,
        "num_layers": num_blocks_vit,
    }

    text_transf_options = {
        "num_heads": num_heads_text,
        "num_blocks": num_blocks_text,
        "embed_dims": embed_dim,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
    }

    saved_model_path = Path("models/ViT_Text_Tranf.pt")
    model = TransformersSingleTextModel(vit_options, text_transf_options)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    model.to(device)

    # Dataloaders and collate_fn
    text_pipeline = lambda x: [vocab_[token] for token in tokenizer.tokenize(x)]

    def collate_batch(batch):
        img_list, text_list = [], []
        for img, _text in batch:
            # Protect against larger sequencies
            # -2 to account for BOS and EOS tokens
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
        )

    ## Careful not to shuffle!
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    ### Define a save name for the embeddings. Should be descriptive
    text_save_path = Path("models/inference/embeddings") / "ViT_Text"
    text_save_path.mkdir(exist_ok=True, parents=True)

    img_save_path = text_save_path / f"embeddings_{cust_name}__img.pt"
    text_save_path = text_save_path / f"embeddings_{cust_name}__text.pt"

    # Try a small part of the dataset
    text_embeddings_stacked = torch.zeros((1, embed_dim)).to(device)
    img_embeddings_stacked = torch.zeros((1, embed_dim)).to(device)

    pbar = tqdm(train_loader)
    for batch in pbar:
        with torch.no_grad():
            img, text = batch

            _, _, text_embeddings, img_embeddings = model(
                img, text
            )  # Careful, bc I have messed up the order in Bert

            text_embeddings_stacked = torch.vstack(
                [text_embeddings_stacked, text_embeddings]
            )

            img_embeddings_stacked = torch.vstack(
                [img_embeddings_stacked, img_embeddings]
            )

    pbar = tqdm(val_loader)
    for batch in pbar:
        with torch.no_grad():
            img, text = batch

            _, _, text_embeddings, img_embeddings = model(
                img, text
            )  # Careful, bc I have messed up the order in Bert

            text_embeddings_stacked = torch.vstack(
                [text_embeddings_stacked, text_embeddings]
            )

            img_embeddings_stacked = torch.vstack(
                [img_embeddings_stacked, img_embeddings]
            )

    # print(text_embeddings_stacked.shape)
    # print(text_embeddings_stacked[0])
    # print(text_embeddings_stacked[1:].shape)
    # print(text_embeddings_stacked[1:][0])

    torch.save(text_embeddings_stacked[1:], text_save_path)
    torch.save(img_embeddings_stacked[1:], img_save_path)


if __name__ == "__main__":
    main()
