import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchtext import data, vocab, datasets
from torchtext.data import TabularDataset

from src.data.make_dataset import DatasetRecipes

import spacy

VOCAB_SIZE = 50000

TEXT = data.Field(
    lower=True, include_lengths=True, batch_first=True, truncate_first=False
)

fields = {"Instructions": ("instructions", TEXT)}

data_path = "data/processed/test/recipes.csv"
custom_examples = TabularDataset(path=data_path, format="csv", fields=fields)

# Build the vocabulary
TEXT.build_vocab(custom_examples, vectors="glove.6B.100d")
# Build the vocabulary
# TEXT.build_vocab(custom_examples, max_size=50000, min_freq=2)


index = 0
text = custom_examples[index].instructions

print(custom_examples[index].instructions)
print(TEXT.vocab.vectors[text])

# data_path = "data/processed/test"

# spacy_en = spacy.load("en")


# def tokenizer(input_text):
#     return [token.text for token in spacy_en.tokenizer(input_text)]

# titles_field = data.Field(lower=True, use_vocab=True, include_lengths=True, batch_first=True, tokenize=tokenizer)
# ingredients_field = data.Field(lower=True, use_vocab=True, include_lengths=True, batch_first=True, tokenize=tokenizer)
# instructions_field  = data.Field(lower=True, use_vocab=True, include_lengths=True, batch_first=True, tokenize=tokenizer)


# dataset = DatasetRecipes(data_path, titles_field, ingredients_field, instructions_field)
