import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from clip import model

from PIL import Image
from pathlib import Path
from src.data.make_dataset import DatasetRecipes
import wandb


BATCH_SIZE = 32
EPOCHS = 100  # 300 looks best up to now
MAX_SEQ_LEN = 77


def main():
    # hyperparams
    lr = 5e-5
    save_per_n_epochs = 10  # if None, save every epoch
    #

    logger = logging.getLogger(__name__)

    wandb.init(project=f"ViT_Text_Transf")
    wandb.config = {"learning_rate": lr, "epochs": EPOCHS, "batch_size": BATCH_SIZE}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    load_path = "ViT-B/32"
    model, preprocess = clip.load(load_path, device=device, jit=False)
    # model.context_length = MAX_SEQ_LEN

    train_data_path = r"data/processed/train"
    val_data_path = r"data/processed/validation"

    def collate_batch(batch):
        img_list, text_list = [], []
        for img, _text in batch:
            # img = torch.tensor(img["pixel_values"][0])

            if len(_text) > MAX_SEQ_LEN:
                _text = _text[:MAX_SEQ_LEN]

            # processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            # processed_text = torch.tensor(
            #     tokenizer(_text)["input_ids"], dtype=torch.int64
            # )

            processed_text = clip.tokenize(_text, context_length=MAX_SEQ_LEN)

            text_list.append(processed_text)
            img_list.append(img.unsqueeze(0))

        # text_list[0] = nn.ConstantPad1d((0, MAX_SEQ_LEN - text_list[0].shape[0]), 0)(
        #     text_list[0]
        # )
        # padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        return (
            torch.cat(img_list, axis=0).to(device),
            torch.cat(text_list, axis=0).to(device),
        )

    columns = ["Title"]
    train_dataset = DatasetRecipes(
        data_path=train_data_path, transformations=preprocess, columns=columns
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch
    )

    val_dataset = DatasetRecipes(
        data_path=val_data_path, transformations=preprocess, columns=columns
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch
    )

    # https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    if device == "cpu":
        model.float()
    # else :
    #     clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_dataloader)
        training_accuracy = 0
        training_loss_ = 0
        for batch in pbar:
            optimizer.zero_grad()

            images, texts = batch

            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            # print("Logits per image")
            # print(logits_per_image)
            # print("*" * 10)
            # print(logits_per_text)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2
            total_loss.backward()

            # if device == "cpu":
            #     optimizer.step()
            # else:
            #     convert_models_to_fp32(model)
            #     optimizer.step()
            #     clip.model.convert_weights(model)

            optimizer.step()

            pbar.set_description(
                f"Epoch {epoch}/{EPOCHS}, Loss: {total_loss.item():.4f}"
            )

            training_loss_ += total_loss.item()

            probs = logits_per_image.softmax(dim=-1)
            preds = torch.argmax(probs, dim=-1)

            training_accuracy += (preds == ground_truth).sum().item()

        # print(f"Training accuracy: {accuracy/len(train_dataset)}")

        # Eval loop
        model.eval()
        pbar = tqdm(val_dataloader)
        val_accuracy = 0
        val_loss = 0
        for batch in pbar:
            with torch.no_grad():
                images, texts = batch

                images = images.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model(images, texts)

                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=device
                )

                total_loss = (
                    loss_img(logits_per_image, ground_truth)
                    + loss_txt(logits_per_text, ground_truth)
                ) / 2

                val_loss += total_loss.item()

                probs = logits_per_image.softmax(dim=-1)
                preds = torch.argmax(probs, dim=-1)

                val_accuracy += (preds == ground_truth).sum().item()

        # print(f"Validation accuracy: {accuracy/len(val_dataset)}")

        wandb.log(
            {
                "training_loss": training_loss_ / (len(train_dataset)) * BATCH_SIZE,
                "training_accuracy": training_accuracy / (len(train_dataset)),
                "validation_loss": val_loss / len(val_dataset) * BATCH_SIZE,
                "validation_accuracy": val_accuracy / len(val_dataset),
            }
        )

        # save model
        if save_per_n_epochs:
            if epoch % save_per_n_epochs == 0 and epoch > 1:
                torch.save(
                    {
                        # 'epoch': epoch,
                        "model_state_dict": model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': total_loss,
                    },
                    f"models/clip_finetuned.pt",
                )
                torch.save(model, f"models/clip_finetuned_full.pt")
        # else:
        #     torch.save(
        #         {
        #             # 'epoch': epoch,
        #             "model_state_dict": model.state_dict(),
        #             # 'optimizer_state_dict': optimizer.state_dict(),
        #             # 'loss': total_loss,
        #         },
        #         f"models/clip_finetuned.pt",
        #     )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
