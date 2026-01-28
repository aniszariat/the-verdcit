import torch
import torch.nn as nn
from activationFunction import FeedForward
from gpt_config import GPT_CONFIG_124M

# let’s initialize a new FeedForward module with a token embedding size of 768 and
# feed it a batch input with two samples and three tokens each
ffn = FeedForward(GPT_CONFIG_124M)
# Creates sample input with batch dimension 2
x = torch.rand(2, 3, 768)
out = ffn(x)
print("ffn shape:", out.shape)
