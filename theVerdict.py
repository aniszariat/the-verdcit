# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:45:42 2026

@author: zaria
"""

import re
import tiktoken
import dataLoaderProcess as dlp
import torch

print("*****")
print("Tokenizing text")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
# print(raw_text[:125])

"""
splits text into individual words and punctuation characters
""" 

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# This print statement outputs 4690, which is the number of tokens in this text (without whitespaces)

print("number of tokens:",len(preprocessed))

# print the 1st 30 token from the list
# print(preprocessed[:30])

"""
we add an <|unk|> token to represent new and unknown words that were not part of the training data and thus not part of the existing vocabulary. 
we add an <|endoftext|> token that we can use to separate two unrelated text sources.
"""
preprocessed.extend(["<|endoftext|>", "<|unk|>"])
"""
let’s create a list of all unique tokens and sort them alphabetically
to determine the vocabulary size:
"""
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("vocabulary size:",vocab_size)


# we create the vocabulary and print its first 51 entries
vocab = {token:integer for integer,token in enumerate(all_words)}

print("\n\n*****")
print("Byte pair encoding")
"""
instantiate the BPE tokenizer fromtiktoken as follows:
"""
tokenizer = tiktoken.get_encoding("gpt2")

# Data sampling with a sliding window
print("\n\n*****")
print("Data sampling with a sliding window")

"""
Let’s implement a data loader that fetches the input–target
pairs from the training dataset using a sliding
window approach. 
"""
"""
we will tokenize the whole “The Verdict” short story using the BPE tokenizer:
"""
"""
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print("BPE tokens: ",len(enc_text))
"""
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
""""""

max_length = 4
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# instantiate the data loader
dataloader = dlp.DataLoaderClass.create_dataloader_v1(
    raw_text,
    batch_size=8, 
    max_length=max_length, 
    stride=max_length, 
    shuffle=False)
data_iter = iter(dataloader)
"""
#1
first_batch = next(data_iter)
print(first_batch)
#1 Converts dataloader into a Python iterator to fetch the next entry via Python’s built-in next() function

second_batch = next(data_iter)
print(second_batch)
"""

"""
Let’s look briefly at how we can use the data loader to sample with a batch size greater than 1:
"""
"""
dataloader = dlp.DataLoaderClass.create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\\n", targets)
"""
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape: ", inputs.shape)

# use the embedding layer to embed these token IDs into 256-dimensional vectors:
token_embeddings = token_embedding_layer(inputs)
print("Token embeddings shape: ",token_embeddings.shape)

# create another embedding layer that has the same embedding dimension as the token_embedding_ layer
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Position embeddings shape: ",pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)