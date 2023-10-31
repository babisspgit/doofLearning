from collections import Counter, OrderedDict
import re

from torch.utils.data import Dataset

from torchtext.vocab import vocab


from typing import Callable

import torchtext




def tokenizer(text:str)->list:
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized


def get_vocab(dataset:Dataset, tokenizer:Callable)->torchtext.vocab.Vocab:
    '''
    From:
    https://github.com/rasbt/machine-learning-book/blob/main/ch15/ch15_part2.py
    '''
        
    # Create the iterable with words and counts
    token_counts = Counter()

    for _, line in dataset:
        tokens = tokenizer(line)
        token_counts.update(tokens)
    
    sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    vocab_ = vocab(ordered_dict)

    vocab_.insert_token("<pad>", 0)
    vocab_.insert_token("<unk>", 1)
    vocab_.set_default_index(1)

    return vocab_

