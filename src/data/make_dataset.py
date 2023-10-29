# -*- coding: utf-8 -*-
import logging
from pathlib import Path

# from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from torchvision import transforms
import torchtext


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

    # def tokenize(self, input_text):
    #     TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
    #     TEXT.build_vocab(input_text, max_size=self.VOCAB_SIZE - 2)
    #     return input_text

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


def main(input_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    dataset = DatasetRecipes(input_filepath)

    i = 10
    a = dataset[i]

    print(a)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    input_filepath = r"data/raw/archive"
    output_filepath = r"data/processed"
    # load_dotenv(find_dotenv())

    main(input_filepath)
