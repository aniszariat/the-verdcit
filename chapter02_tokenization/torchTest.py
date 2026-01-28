#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 12:11:21 2026

@author: dell
"""
import torch


# Set the seed to a specific integer value (e.g., 42)
# This affects both CPU and CUDA devices
torch.manual_seed(42)

# Generate a tensor with random numbers
tensor_1 = torch.rand(3)
print(f"Tensor 1: {tensor_1}")

# Reset the seed to the SAME value
torch.manual_seed(42)

# Generate a second tensor using the same random operation
tensor_2 = torch.rand(3)
print(f"Tensor 2: {tensor_2}")

# The output of both tensors will be identical
assert torch.equal(tensor_1, tensor_2)
print("Tensors are identical: True")

"""
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))
"""
