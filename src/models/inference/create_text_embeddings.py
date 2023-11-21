import torch
import torch.nn as nn

from torchvision import transforms

from pathlib import Path
from tqdm import tqdm

from src.models.models import TransformersSingleTextModel
from src.utils.vocab_build import CustomTokenizer, get_vocab

from src.data.make_dataset import DatasetRecipesSep

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

    scheduler_step = config.scheduler_step
    scheduler_gamma = config.scheduler_gamma

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

    # (Train) data path
    train_path = data_path / "train"
    train_dataset = DatasetRecipesSep(data_path=train_path)

    # Vocab and tokenizers
    tokenizer = CustomTokenizer()

    vocab_, MAX_SEQ_LEN = get_vocab(train_dataset, tokenizer=tokenizer.tokenize)

    VOCAB_SIZE = len(vocab_)

    saved_model_path = Path("models/ViT_Text_Tranf.pt")

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

    model = TransformersSingleTextModel(vit_options, text_transf_options)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()


if __name__ == "__main__":
    main()
