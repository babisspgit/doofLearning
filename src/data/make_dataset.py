# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import random

# from dotenv import find_dotenv, load_dotenv
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

from torchvision import transforms
import torchtext

from tqdm import tqdm


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DatasetRecipes(Dataset):
    def __init__(self, data_path, transformations=None):
        super(DatasetRecipes, self).__init__()

        self.MAX_SEQ_LEN = 512
        self.VOCAB_SIZE = 50_000

        csv_path = (
            Path(data_path)
            / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
        )

        self.recipes_df = pd.read_csv(csv_path)  # For now load everything
        self.image_path = Path(data_path) / "Food Images" / "Food Images"

        if not transformations:
            self.transformations = transforms.Compose(
                [
                    # transforms.Resize(200,200),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            self.transformations = transformations

    def __len__(self):
        return len(self.recipes_df)

    def __getitem__(self, idx):
        data_point = self.recipes_df.iloc[idx]

        # Prepare the text data

        title = data_point.Title

        # Would it be better if we dropeed the sq. brackets at the beg and end?
        ingredients = data_point.Cleaned_Ingredients  # Look the same as Ingredients
        instruction = data_point.Instructions

        # Prepare the image
        image_name = data_point.Image_Name + ".jpg"
        image_path = self.image_path / image_name

        try:
            img = Image.open(image_path)
        except FileNotFoundError as e:
            print(f"Image index: {idx}")
            return None, None

        return self.transformations(img), title + ingredients + instruction


def img_exists(img_path):
    try:
        Image.open(img_path)
    except FileNotFoundError:
        return False

    return True


def get_good_idx(raw_df, img_dir):
    df = raw_df.copy()

    good_idx = []

    pbar = tqdm(range(len(df)), desc="Cleaning raw data")

    for i in pbar:
        row = df.iloc[i]
        # Test if text fields are OK
        if not row.Title:
            continue
        if not row.Instructions:
            continue
        if not row.Cleaned_Ingredients or row.Cleaned_Ingredients == "['']":
            continue

        # Test if corr. image exists
        img_path = img_dir / (f"{row.Image_Name}.jpg")
        if not img_exists(img_path):
            continue

        good_idx.append(i)

    return good_idx


def main(input_filepath, output_path, seed=42):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Set the seed for splitting the datasets
    set_seed(seed)

    # Define raw paths
    raw_csv_path = (
        input_filepath
        / "archive"
        / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    )

    raw_images_path = input_filepath / "archive" / "Food Images" / "Food Images"
    raw_df = pd.read_csv(raw_csv_path)

    good_idx = get_good_idx(raw_df, raw_images_path)

    # Create the processed dir
    logger.info("Creating folders")
    processed_train = output_path / "train"
    processed_validation = output_path / "validation"
    processed_test = output_path / "test"

    Path.mkdir(processed_train, exist_ok=True, parents=True)
    Path.mkdir(processed_validation, exist_ok=True, parents=True)
    Path.mkdir(processed_test, exist_ok=True, parents=True)

    # From the good idx, split the datasets in train val test
    splits = [0.8, 0.1, 0.1]

    assert (
        np.abs(splits[0] + splits[1] + splits[-1] - 1) < 1e-6
    ), "The splits should add to 1"

    clean_df = raw_df.iloc[good_idx, :]

    print(len(raw_df), len(clean_df))

    train, validate, test = np.split(
        clean_df.sample(frac=1, random_state=42),
        [int(splits[0] * len(clean_df)), int((splits[0] + splits[1]) * len(clean_df))],
    )

    print(len(train), len(validate), len(test))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    input_filepath = Path(r"data/raw")
    output_filepath = Path(r"data/processed")
    # load_dotenv(find_dotenv())

    main(input_filepath, output_filepath)
