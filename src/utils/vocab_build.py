from collections import Counter, OrderedDict

from torch.utils.data import Dataset
from typing import Callable

from torchtext.vocab import vocab
import torchtext

import spacy


from pathlib import Path
from tqdm import tqdm
import pickle


from src.data.make_dataset import DatasetRecipes


class CustomTokenizer:
    def __init__(self):
        self.tokenizer = spacy.load(
            "en_core_web_sm",
        )

    def tokenize(self, text):
        text = text.replace("[", "").replace("]", "").replace("'", "")
        # could also .lower()

        tokens_list = self.tokenizer(text)

        tokens_list = [i.text for i in tokens_list]

        return tokens_list


def get_vocab(
    dataset: Dataset,
    tokenizer: Callable,
    vocab_save_path: str = "models/custom_vocab.pkl",
    load_saved: bool = True,
) -> torchtext.vocab.Vocab:
    """
    From:
    https://github.com/rasbt/machine-learning-book/blob/main/ch15/ch15_part2.py
    """

    # Where to look for or save the vocab
    vocab_save_path = Path(vocab_save_path)

    if vocab_save_path.exists() and load_saved:
        print("Found custom vocab")
        with open(vocab_save_path, "rb") as fin:
            a = pickle.load(fin)
            return a["vocab"], a["max_seq_len"]

    # Create the iterable with words and counts
    token_counts = Counter()

    max_seq_len = 0

    for _, line in tqdm(dataset, desc="Building vocabulary", leave=False):
        tokens = tokenizer(line)
        if len(tokens) > max_seq_len:
            max_seq_len = len(tokens)
        token_counts.update(tokens)

    sorted_by_freq_tuples = sorted(
        token_counts.items(), key=lambda x: x[1], reverse=True
    )

    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
    vocab_ = vocab(ordered_dict, specials=specials)

    vocab_.set_default_index(vocab_["<unk>"])

    with open(vocab_save_path, "wb") as fout:
        pickle.dump({"vocab": vocab_, "max_seq_len": max_seq_len}, fout)

    print("Saved custom vocab")

    return vocab_, max_seq_len


def generate_vocabulary(dataset: Dataset, output_name: str) -> bool:
    tokenizer = CustomTokenizer()
    vocab_, MAX_SEQ_LEN = get_vocab(
        dataset,
        tokenizer=tokenizer.tokenize,
        vocab_save_path=output_name,
        load_saved=False,
    )

    if (not vocab_) or (len(vocab_) < 1) or MAX_SEQ_LEN < 1:
        return False

    return True


if __name__ == "__main__":
    # build a dictionary using the
    # training dataset only(?)
    # so that it does not happen
    # during training

    data_path = Path("data/processed/train")
    dataset = DatasetRecipes(data_path=data_path)

    output_name = "models/custom_vocab.pkl"
    success = generate_vocabulary(dataset=dataset, output_name=output_name)

    if not success:
        print("Failed to generate dictionary")
