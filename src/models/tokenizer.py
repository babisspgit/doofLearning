import re
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab
from torchtext import vocab


def tokenizer(raw_text: str) -> list:
    """
    Transforms a string input to a list of strings, aka tokens
    Preprocesses-removes unwanted characters
    """

    # TODO Decide which characters to remove, replace with empty string, eg. '[' or ']'
    text = raw_text.replace("[", "")
    text = text.replace("]", "")
    text = text.lower()

    # There exist also some special chaarcters , like the unicode for 3/4. Let them be?

    # TODO Decide what to do with punctuation, like ','. I let them be for now.

    tokenized_text = text.split()

    return tokenized_text


def main(dataset_):
    token_counts = Counter()

    for _, text_ in dataset_:
        if not text_:
            continue
        tokens = tokenizer(text_)
        token_counts.update(tokens)

    print(f"Vocab length: {len(token_counts)}")

    sorted_counted_words = sorted(
        token_counts.items(), key=lambda x: x[1], reverse=True
    )
    ord_dict = OrderedDict(sorted_counted_words)
    vocab_ = vocab.build_vocab_from_iterator(ord_dict)


if __name__ == "__main__":
    # Only to test the tokenizer! Does not save anything for the model to see!

    from src.data.make_dataset import DatasetRecipes

    csv_path = r"data/processed/validation"
    dataset = DatasetRecipes(csv_path)
    main(dataset)
