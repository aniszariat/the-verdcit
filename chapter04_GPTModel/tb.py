import torch
from transformerBlockClass import TransformerBlock
from gpt_config import GPT_CONFIG_124M

# let’s instantiate a transformer block and feed it some sample data

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
# Creates sample input of shape [batch_size, num_tokens, emb_dim]
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
