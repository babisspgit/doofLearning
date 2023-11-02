import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from pathlib import Path
import pandas as pd
from PIL import Image

import clip
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizerFast


class RecipesDataset(Dataset):
    def __init__(self, data_path, processor):
        super().__init__()
        
        csv_path = Path(data_path) / "recipes.csv"

        self.recipes_df = pd.read_csv(csv_path)
        self.image_path = Path(data_path) / "images"

        self.processor = processor

    def __len__(self):
        return len(self.recipes_df)
    
    def __getitem__(self, idx):
        data_point = self.recipes_df.iloc[idx]

        # Prepare the text data
        title = data_point.Title
        ingredients = data_point.Cleaned_Ingredients
        instructions = data_point.Instructions

        text = title # + ingredients + instructions
        
        processed_text = clip.tokenize(text, context_length=77)

        # print(f"Processed text:\n{processed_text}")

        # Prepare the image
        image_name = data_point.Image_Name + ".jpg"
        image_path = self.image_path / image_name

        try:
            img = Image.open(image_path)
        except FileNotFoundError as e:
            print(e)
            print(f"Image index: {idx}")
            return None, None
        
        # processed_img = self.img_processor(img)

        return  self.processor(img), processed_text



def main():

    batch_size = 2
    num_epochs = 30
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # pretrained:
    # processor: "openai/clip-vit-base-patch32"
    # model: "openai/clip-vit-base-patch32"

    pretrained_str = "openai/clip-vit-base-patch32"

    processor = CLIPImageProcessor.from_pretrained(pretrained_str)

    
    train_dataset = RecipesDataset('data/processed/train', processor)


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)


    model = CLIPModel.from_pretrained(pretrained_str)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    

    
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader)
        for batch in pbar:

            optimizer.zero_grad()


            

            images, texts = batch
            images = images['pixel_values'][0]
        
            # images= images.to(device)
            # texts = texts.to(device)

            print(images.shape)
            print(texts.shape)

            # Forward pass
            logits_per_image, logits_per_text = model(pixel_values=images, input_ids=texts)


            return

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

            # Backward pass
            total_loss.backward()
            optimizer.step()
           

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
    



if __name__=='__main__':
    
    options = {

    }
    
    main()