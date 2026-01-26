import tiktoken
import torch
from DummyGPTModelClass import DummyGPTModel
import torch.nn as nn

# The configuration of the small GPT-2 model via the following Python dictionary:
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}

# we tokenize a batch consisting of two text inputs for the GPT model using the tiktoken tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# we initialize a new 124-million-parameter DummyGPTModel instance and feed it the tokenized batch:
""" 
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
"""
# implement layer normalization to improve the stability and efficiency of neural network training
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
# 1 Creates two training examples with five dimensions (features) each
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# let’s examine the mean and variance:
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)