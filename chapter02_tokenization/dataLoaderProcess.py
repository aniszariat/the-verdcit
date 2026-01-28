#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:27:06 2026

@author: dell
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
"""
A dataset for batched inputs and targets
"""
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        #1 Tokenizes the entire text
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            #2 Uses a sliding window to chunk the book into overlapping sequences of max_length
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        #3 Returns the total number of rows in the dataset
        return len(self.input_ids)
    def __getitem__(self, idx):
        #4 Returns a single row from the dataset
        return self.input_ids[idx], self.target_ids[idx]
    

"""
A data loader to generate batches with input-with pairs
"""
class DataLoaderClass: 
    def create_dataloader_v1(
            txt, 
            max_length=256,
            batch_size=4, 
            stride=128, 
            shuffle=True, 
            drop_last=True,
            num_workers=0):
        #1 Initializes the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        #2 Creates dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        #3 drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
        #4 The number of CPU processes to use for preprocessing
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,        #3
            num_workers=num_workers        #4
            )
        return dataloader