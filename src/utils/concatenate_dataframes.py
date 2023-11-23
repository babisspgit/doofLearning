import pandas as pd
from pathlib import Path


def main():
    """
    Since I have embeddings for the train and validation datasets, I need to
    concatenate their text data .csv's in the corresponding order.
    This is done here and stored in the
    "models/inference/"  folder
    """

    # Create the dir to save the .csv in
    save_path = Path("models/inference")
    save_path.mkdir(exist_ok=True, parents=True)
    save_path = save_path / "recipes.csv"

    # Load the dataframes
    recipes_train_path = Path("data/processed") / "train" / "recipes.csv"
    recipes_validation_path = Path("data/processed") / "validation" / "recipes.csv"

    train_df = pd.read_csv(recipes_train_path)
    val_df = pd.read_csv(recipes_validation_path)

    ## Remember to concatenate with the same order as they are added in the embeddings tensor!!
    ## This can be seen in "models/inference/create_text_embeddings.py" file
    ## It is the order the differen pbars appear in the file

    recipes_df = pd.concat([train_df, val_df], ignore_index=True)

    assert (len(train_df) + len(val_df)) == len(
        recipes_df
    ), "Concatenated dataframes do not contain the same number of rows"
    assert len(train_df.columns) == len(
        recipes_df.columns
    ), "Concatenated dataframes do not have the same number of columns"

    recipes_df.to_csv(save_path, index=0)

    print("Done!")


if __name__ == "__main__":
    main()
