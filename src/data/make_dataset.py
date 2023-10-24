# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class DatasetTitleOnly(Dataset):
    def __init__(self, image_path, csv_path, transformations=None):
        super(DatasetTitleOnly, self).__init__()

        self.recipes_df = pd.read_csv(csv_path) # For now load everything

        if transformations:
            self.transformations = transformations

    def __len__(self):
        return len(self.recipes_df)

    def __getitem__(self, idx):
        img = Image.open(self.image_names_list[idx])

        # Only get the title
        recipe_title = self.recip




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
