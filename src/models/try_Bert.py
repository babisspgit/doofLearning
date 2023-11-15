import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from transformers import BertModel, BertTokenizer

from src.models.models import Transformers_Bert

from src.data.make_dataset import DatasetRecipes

from tqdm import tqdm
from pathlib import Path
import random
import numpy as np


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # hparams

    seed = 0

    set_seed(seed=seed)

    lr = 1e-4
    batch_size = 32
    n_epochs = 50
    save_per_n_epochs = 10

    scheduler_step = 10
    scheduler_gamma = 0.5

    # data path(s)
    data_path = Path("data/processed")

    # Unpack experiment specific params

    embed_dim = 512

    # ViT
    image_dims = [256, 256]
    patch_dims = (32, 32)
    num_heads_vit = 2
    num_blocks_vit = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_dims),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_set = DatasetRecipes(data_path / "train", [], transformations=train_transform)
    val_set = DatasetRecipes(data_path / "validation", [])

    MAX_SEQ_LEN = 512  # Fixed for Bert
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def collate_batch(batch):
        img_list, text_list, attention_masks_list = [], [], []
        for img, _text in batch:
            # Protect against larger sequencies
            # -2 to account for [CLS] and [SEP] tokens
            if len(_text) > MAX_SEQ_LEN - 2:
                _text = _text[: MAX_SEQ_LEN - 2]

            processed_text = tokenizer.tokenize(_text)

            # Add special characters, like in
            # https://jalammar.github.io/illustrated-bert/
            processed_text = ["[CLS]"] + processed_text + ["[SEP]"]
            pads = ["[PAD]" for _ in range(abs(MAX_SEQ_LEN - len(processed_text)))]
            processed_text = processed_text + pads

            attention_mask = [0 if i == "[PAD]" else 1 for i in processed_text]

            processed_text = tokenizer.convert_tokens_to_ids(processed_text)

            processed_text = torch.tensor(processed_text, dtype=torch.int64)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)

            text_list.append(processed_text.unsqueeze(0))
            attention_masks_list.append(attention_mask.unsqueeze(0))

            # Since the batching is manual, in this fcn, I need to add batch dim
            img_list.append(img.unsqueeze(0))

        return (
            torch.cat(img_list, axis=0).to(device),
            torch.cat(text_list, axis=0).to(device),
            torch.cat(attention_masks_list, axis=0).to(device),
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_batch, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=collate_batch, shuffle=False
    )

    vit_options = {
        "img_dims": image_dims,
        "channels": 3,
        "patch_sizes": patch_dims,
        "embed_dim": embed_dim,
        "num_heads": num_heads_vit,
        "num_layers": num_blocks_vit,
    }

    # model = BertModel.from_pretrained("bert-base-uncased")
    model = Transformers_Bert(vit_options)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_fcn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        pbar = tqdm(train_loader)
        model.train()

        for batch in pbar:
            img, in_idx, att_masks = batch

            # Use contrastive loss
            labels = torch.arange(img.shape[0])  # should be eq to batch_size
            labels = labels.to(device)

            _, _, logits_per_text, logits_per_image = model(
                img, in_idx, attention_mask=att_masks
            )

            loss_img = loss_fcn(logits_per_image, labels)
            loss_text = loss_fcn(
                logits_per_text, labels
            )  # maybe logits transpose? should not matter at all

            loss = (loss_img + loss_text) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
