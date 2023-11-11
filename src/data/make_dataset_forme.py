# Importing necessary libraries
import logging
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import torchtext
from tqdm import tqdm

# Set random seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Define a custom dataset class for handling images and text data
class DatasetRecipes(Dataset):
    def __init__(self, data_path, transformations=None):
        super(DatasetRecipes, self).__init__()

        # Constants for maximum sequence length and vocabulary size
        self.MAX_SEQ_LEN = 512
        self.VOCAB_SIZE = 50_000

        # Read CSV file containing recipes and image mappings
        csv_path = Path(data_path) / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
        self.recipes_df = pd.read_csv(csv_path)
        self.image_path = Path(data_path) / "Food Images" / "Food Images"

        # Define default image transformations if not provided
        if not transformations:
            self.transformations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transformations = transformations

    def __len__(self):
        return len(self.recipes_df)

    def __getitem__(self, idx):
        # Access data for a specific index
        data_point = self.recipes_df.iloc[idx]

        # Extract text data: title, ingredients, and instructions
        title = data_point.Title
        ingredients = data_point.Cleaned_Ingredients
        instruction = data_point.Instructions

        # Prepare the image data
        image_name = data_point.Image_Name + ".jpg"
        image_path = self.image_path / image_name

        # Attempt to open the image; if not found, return None
        try:
            img = Image.open(image_path)
        except FileNotFoundError as e:
            print(f"Image index: {idx}")
            return None, None

        # Apply transformations to the image and concatenate text data
        return self.transformations(img), title + ingredients + instruction

# Function to check if an image exists at the specified path
def img_exists(img_path):
    try:
        Image.open(img_path)
    except FileNotFoundError:
        return False
    return True

# Filter and return indices of entries with valid text and image paths
def get_good_idx(raw_df, img_dir):
    df = raw_df.copy()
    good_idx = []
    pbar = tqdm(range(len(df)), desc="Cleaning raw data")
    for i in pbar:
        row = df.iloc[i]
        # Test if text fields are OK
        if not row.Title or not row.Instructions or not row.Cleaned_Ingredients or row.Cleaned_Ingredients == "['']":
            continue
        # Test if corresponding image exists
        img_path = img_dir / (f"{row.Image_Name}.jpg")
        if not img_exists(img_path):
            continue
        good_idx.append(i)
    return good_idx

# Primary function for data processing and dataset splitting
def main(input_filepath, output_path, seed=42):
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    # Set the seed for data splitting
    set_seed(seed)

    # Define paths for raw dataset and images
    raw_csv_path = input_filepath / "archive" / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    raw_images_path = input_filepath / "archive" / "Food Images" / "Food Images"

    # Read the raw dataset
    raw_df = pd.read_csv(raw_csv_path)

    # Get indices of entries with valid text and image paths
    good_idx = get_good_idx(raw_df, raw_images_path)

    # Create directories for processed data
    logger.info("Creating folders")
    processed_train = output_path / "train"
    processed_validation = output_path / "validation"
    processed_test = output_path / "test"
    Path.mkdir(processed_train, exist_ok=True, parents=True)
    Path.mkdir(processed_validation, exist_ok=True, parents=True)
    Path.mkdir(processed_test, exist_ok=True, parents=True)

    # Split the data into train, validation, and test sets
    splits = [0.8, 0.1, 0.1]
    assert np.abs(splits[0] + splits[1] + splits[-1] - 1) < 1e-6, "The splits should add to 1"
    clean_df = raw_df.iloc[good_idx, :]
    train, validate, test = np.split(clean_df.sample(frac=1, random_state=42),
                                     [int(splits[0] * len(clean_df)), int((splits[0] + splits[1]) * len(clean_df))])
    print(len(train), len(validate), len(test))

if __name__ == "__main__":
    # Configure logging settings
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define input and output file paths
    input_filepath = Path(r"data/raw")
    output_filepath = Path(r"data/processed")

    # Call the main function to process the data
    main(input_filepath, output_filepath)
