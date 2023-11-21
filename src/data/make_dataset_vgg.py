import logging
from pathlib import Path

import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


from torchvision import transforms

from tqdm import tqdm

import skimage.measure


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DatasetRecipesTriplet(Dataset):
    def __init__(
        self, data_path, columns, img_transformations=None
    ):
        super(DatasetRecipesTriplet, self).__init__()

        csv_path = Path(data_path) / "recipes.csv"

        self.recipes_df = pd.read_csv(csv_path)
        self.image_path = Path(data_path) / "images"

        if columns:
            self.columns = columns[:]
        else:
            # Use all
            self.columns = ["Title", "Cleaned_Ingredients", "Instructions"]

        self.transformations = img_transformations
        if not img_transformations:
            self.transformations = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

    def __len__(self):
        return len(self.recipes_df)

    def __getitem__(self, idx):
        data_point = self.recipes_df.iloc[idx]

        # Prepare the text data
        raw_text = ""
        for col in self.columns:
            raw_text += data_point[col] + " "

        # Remove the trailing space
        raw_text = raw_text.strip()

        # Prepare the image
        image_name = data_point.Image_Name + ".jpg"
        image_path = self.image_path / image_name

        try:
            img = Image.open(image_path)
        except FileNotFoundError as e:
            print(f"Image index: {idx}")
            return None, None

        # Get the negative pairs
        neg_idx = torch.randint(len(self.recipes_df), (1,)).item()

        neg_data_point = self.recipes_df.iloc[neg_idx]

        # Prepare the text data
        neg_raw_text = ""
        for col in self.columns:
            neg_raw_text += neg_data_point[col] + " "

        # Remove the trailing space
        neg_raw_text = neg_raw_text.strip()

        # Prepare the image
        neg_image_name = neg_data_point.Image_Name + ".jpg"
        neg_image_path = self.image_path / neg_image_name

        try:
            neg_img = Image.open(neg_image_path)
        except FileNotFoundError as e:
            print(f"Image index: {neg_idx}")
            return None, None

        # Apply image transformations only once
        img = self.transformations(img)
        neg_img = self.transformations(neg_img)

        # Recombine in text and image anchor-pos-neg triplets
        img_dict = {"anchor": img, "positive": raw_text, "negative": neg_raw_text}

        text_dict = {"anchor": raw_text, "positive": img, "negative": neg_img}

        return img_dict, text_dict


class DatasetRecipes(Dataset):
    def __init__(
        self,
        data_path,
        columns,
        transformations=None,
    ):
        super(DatasetRecipes, self).__init__()

        csv_path = Path(data_path) / "recipes.csv"

        self.recipes_df = pd.read_csv(csv_path)
        self.image_path = Path(data_path) / "images"

        if columns:
            self.columns = columns[:]
        else:
            # Use all
            self.columns = ["Title", "Cleaned_Ingredients", "Instructions"]

        self.transformations = transformations
        if not transformations:
            self.transformations = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

    def __len__(self):
        return len(self.recipes_df)

    def __getitem__(self, idx):
        data_point = self.recipes_df.iloc[idx]

        # Prepare the text data
        raw_text = ""
        for col in self.columns:
            raw_text += data_point[col] + " "

        # Remove the trailing space
        raw_text = raw_text.strip()

        # Prepare the image
        image_name = data_point.Image_Name + ".jpg"
        image_path = self.image_path / image_name

        try:
            img = Image.open(image_path)
        except FileNotFoundError as e:
            print(f"Image index: {idx}")
            return None, None

        return self.transformations(img), raw_text


def img_exists(img_path, entropy_lim=4.5):
    try:
        img_ = Image.open(img_path)
        entropy = skimage.measure.shannon_entropy(img_)
        if entropy < entropy_lim:
            return False

    except: #FileNotFoundError:
        return False

    return True


def get_good_idx(raw_df, img_dir):
    df = raw_df.copy()

    good_idx = []

    pbar = tqdm(range(len(df)), desc="Cleaning raw data")

    for i in pbar:
        row = df.iloc[i]
        # Test if text fields are OK
        if not row.Title or not (type(row.Title) == str):
            continue
        if not row.Instructions or not (type(row.Instructions) == str):
            continue
        if (
            not row.Cleaned_Ingredients
            or row.Cleaned_Ingredients == "['']"
            or not (type(row.Cleaned_Ingredients) == str)
        ):
            continue

        # Test if corr. image exists
        img_path = img_dir / (f"{row.Image_Name}.jpg")
        if not img_exists(img_path):
            continue

        good_idx.append(i)

    return good_idx


def save_images(df, root_path, raw_images_path):
    pbar = tqdm(range(len(df)), desc="Saving images")

    images_path = root_path / "images"
    Path.mkdir(images_path, exist_ok=True, parents=True)

    for i in pbar:
        row = df.iloc[i]
        img_path_read = raw_images_path / f"{row.Image_Name}.jpg"
        img_write_path = images_path / f"{row.Image_Name}.jpg"
        if img_write_path.exists():
            continue
        Image.open(img_path_read).convert("RGB").save(img_write_path)


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

    # Create the processed dirs
    logger.info("Creating folders")
    processed_train = output_path / "train"
    processed_validation = output_path / "validation"
    processed_test = output_path / "test"

    Path.mkdir(processed_train, exist_ok=True, parents=True)
    Path.mkdir(processed_validation, exist_ok=True, parents=True)
    Path.mkdir(processed_test, exist_ok=True, parents=True)

    # From the good idx, split the datasets in train val test
    splits = [0.85, 0.1, 0.05]
    # splits = [0.99, 0.01]

    assert (
        np.abs(splits[0] + splits[1] + splits[-1] - 1) < 1e-6
    ), "The splits should add to 1"

    clean_df = raw_df.iloc[good_idx]

    train_df, validate_df, test_df = np.split(
        clean_df.sample(frac=1, random_state=42),
        [int(splits[0] * len(clean_df)), int((splits[0] + splits[1]) * len(clean_df))],
    )

    # train_df, test_df = np.split(
    #     clean_df.sample(frac=1, random_state=42),
    #     [int(splits[0] * len(clean_df))],
    # )

    train_path = processed_train / "recipes.csv"
    train_df.to_csv(train_path.as_posix(), index=False, header=True)

    val_path = processed_validation / "recipes.csv"
    validate_df.to_csv(val_path.as_posix(), index=False, header=True)

    test_path = processed_test / "recipes.csv"
    test_df.to_csv(test_path.as_posix(), index=False, header=True)

    # Handle the images
    save_images(train_df, processed_train, raw_images_path)
    save_images(validate_df, processed_validation, raw_images_path)
    save_images(test_df, processed_test, raw_images_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    input_filepath = Path(r"data/raw")
    output_filepath = Path(r"data/processed")

    main(input_filepath, output_filepath)
