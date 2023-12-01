import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from pathlib import Path
import pandas as pd
from PIL import Image

from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor, CLIPImageProcessor

from src.data.make_dataset import DatasetRecipes
from src.utils.vocab_build import get_vocab


# class RecipesDataset(Dataset):
#     def __init__(self, data_path, processor):
#         super().__init__()

#         csv_path = Path(data_path) / "recipes.csv"

#         self.recipes_df = pd.read_csv(csv_path)
#         self.image_path = Path(data_path) / "images"

#         self.processor = processor

#     def __len__(self):
#         return len(self.recipes_df)

#     def __getitem__(self, idx):
#         data_point = self.recipes_df.iloc[idx]

#         # Prepare the text data
#         title = data_point.Title
#         ingredients = data_point.Cleaned_Ingredients
#         instructions = data_point.Instructions

#         final_text = title  # + ingredients + instructions

#         # print(f"Processed text:\n{processed_text}")

#         # Prepare the image
#         image_name = data_point.Image_Name + ".jpg"
#         image_path = self.image_path / image_name

#         try:
#             img = Image.open(image_path)
#         except FileNotFoundError as e:
#             print(e)
#             print(f"Image index: {idx}")
#             return None, None

#         # processed_img = self.img_processor(img)

#         return self.processor(text=final_text, images=img)

# clip accepts context length of 77.
# check not implemented error in method change_context_size
MAX_SEQ_LEN = 77


def main():
    batch_size = 32  # also 64
    num_epochs = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_str = "openai/clip-vit-base-patch32"

    img_processor = CLIPImageProcessor.from_pretrained(pretrained_str)

    tokenizer_options = {
        "unk_token": "<|unk|>",
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "is_split_into_words": True,
        "add_prefix_space": True,
        "padding": True,
        "return_tensors": "pt",
    }

    tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_str, **tokenizer_options)

    columns = ['Title']

    train_dataset = DatasetRecipes(
        "data/processed/train", transformations=img_processor, columns=columns
    )

    val_dataset = DatasetRecipes(
        "data/processed/validation", transformations=img_processor, columns=columns
    )

    # build vocab, add tokens
    # vocab = get_vocab(train_dataset, tokenizer=tokenizer)

    # Create the collate function, since
    # tokenized sequencies have varying length
    # text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

    def collate_batch(batch):
        img_list, text_list = [], []
        for img, _text in batch:
            img = torch.tensor(img["pixel_values"][0])

            # if len(_text) > MAX_SEQ_LEN:
            #     _text = _text[:MAX_SEQ_LEN]

            # processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            processed_text = torch.tensor(
                tokenizer(_text)["input_ids"], dtype=torch.int64
            )
            text_list.append(processed_text)
            img_list.append(img.unsqueeze(0))

        text_list[0] = nn.ConstantPad1d((0, MAX_SEQ_LEN - text_list[0].shape[0]), 0)(
            text_list[0]
        )
        padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        return (torch.cat(img_list, axis=0).to(device), padded_text_list.to(device))

    # Create a DataLoader using also the collate fcn
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = CLIPModel.from_pretrained(pretrained_str)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader)
        train_accuracy = 0
        model.train()
        for batch in pbar:
            optimizer.zero_grad()

            images, input_seq = batch
            btch_sz, seq_len = input_seq.size()
            if seq_len > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]

            # images= images.to(device)
            # texts = texts.to(device)

            # Forward pass
            outputs = model(pixel_values=images, input_ids=input_seq, return_loss=True)

            # return CLIPOutput(
            # loss=loss,
            # logits_per_image=logits_per_image,
            # logits_per_text=logits_per_text,
            # text_embeds=text_embeds,
            # image_embeds=image_embeds,
            # text_model_output=text_outputs,
            # vision_model_output=vision_outputs,
            # )

            loss = outputs.loss
            logits_per_image = outputs.logits_per_image
            softmaxed_logits = nn.functional.softmax(logits_per_image, dim=1)
            preds = torch.argmax(softmaxed_logits, dim=-1).cpu().detach()
            ground_truths = torch.arange(btch_sz)

            # print(preds.shape, ground_truths.shape)

            # print(logits_per_image)
            # print("*" * 10)
            # print(softmaxed_logits)
            # print("*" * 10)
            # print(preds)
            # print("*" * 10)
            # print(ground_truths)
            # print("*" * 10)

            train_accuracy += (ground_truths == preds).sum().item()

            # Backward pass
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

        print(f"Training accuracy: {train_accuracy/len(train_dataset)}")

        for batch in tqdm(val_loader, desc="Validation:"):
            model.eval()
            val_accuracy = 0
            with torch.no_grad():
                images, input_seq = batch
                btch_sz, seq_len = input_seq.size()
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]

                outputs = model(
                    pixel_values=images, input_ids=input_seq, return_loss=True
                )
                logits_per_image = outputs.logits_per_image
                preds = (
                    torch.argmax(nn.functional.softmax(logits_per_image, dim=1), dim=-1)
                    .detach()
                    .cpu()
                )
                ground_truths = torch.arange(btch_sz)

                accuracy = (ground_truths == preds).sum().item()

                val_accuracy += accuracy

        print(val_accuracy / len(val_dataset))


if __name__ == "__main__":
    options = {}

    main()
