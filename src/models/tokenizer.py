# from python_environment_check import check_packages
import torch
import torch.nn as nn
from torch.utils.data.dataset import random_split
import re
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB

from src.data.make_dataset import DatasetRecipes

MAX_LEN=50

train_dataset = IMDB(split='train')

data_path = 'data/processed/train'
train_dataset = DatasetRecipes(data_path)

torch.manual_seed(1)
# train_dataset, _ = random_split(
#     list(train_dataset), [20000, 5000])



## Step 2: find unique tokens (words)

token_counts = Counter()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized


for _, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)
 
    
print('Vocab-size:', len(token_counts))



## Step 3: encoding each unique token into integers

sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)

vocab = vocab(ordered_dict)

vocab.insert_token("<pad>", 0)
vocab.insert_token("<unk>", 1)
vocab.set_default_index(1)

print([vocab[token] for token in ['this', 'is', 'an', 'example', "Ioannis", "Nojus", "OBST"]])

# Pipeline
device = torch.device('cpu')

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1. if x == 'pos' else 0.





## Step 3-B: wrap the encode and transformation function
def collate_batch(batch):
    
    img_list ,text_list, lengths = [], [], []
    for img, _text in batch:
        processed_text = torch.tensor(text_pipeline(_text), 
                                      dtype=torch.int64)
        
        text_list.append(processed_text)
        img_list.append(img.unsqueeze(0))
        lengths.append(processed_text.size(0))
    lengths = torch.tensor(lengths)
    text_list[0] = nn.ConstantPad1d((0, MAX_LEN - text_list[0].shape[0]), 0)(text_list[0])
    padded_text_list = nn.utils.rnn.pad_sequence(
        text_list, batch_first=True)
    return torch.cat(img_list, axis=0).to(device), padded_text_list.to(device), lengths.to(device)




## Take a small batch

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
img, text_batch, length_batch = next(iter(dataloader))

# print(img.shape)
# # print(text_batch)
# print(length_batch)
# print(text_batch.shape)

i=0
for batch in dataloader:
    if i==5:
        break
    img, text_batch, length_batch = batch
    print(img.shape)
    # print(text_batch)
    print(length_batch)
    print(text_batch.shape)

    print("*"*10)
    
    i +=1




## Step 4: batching the datasets

# batch_size = 32  

# train_dl = DataLoader(train_dataset, batch_size=batch_size,
#                       shuffle=True, collate_fn=collate_batch)





# embedding = nn.Embedding(num_embeddings=10, 
#                          embedding_dim=3, 
#                          padding_idx=0)
 
# # a batch of 2 samples of 4 indices each
# text_encoded_input = torch.LongTensor([[1,2,4,5],[4,3,2,0]])
# print(embedding(text_encoded_input))