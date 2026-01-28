import torch
import torch.nn as nn
from activationFunction import FeedForward

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
# let’s initialize a new FeedForward module with a token embedding size of 768 and
# feed it a batch input with two samples and three tokens each
ffn = FeedForward(GPT_CONFIG_124M)
# Creates sample input with batch dimension 2
x = torch.rand(2, 3, 768)
out = ffn(x)
print("ffn shape:", out.shape)
