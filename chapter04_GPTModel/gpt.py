import tiktoken
import torch

# import torch.nn as nn
# from DummyGPTModelClass import DummyGPTModel
# from layerNormalizationClass import LayerNorm
# from gpt_config import GPT_CONFIG_124M
from GPTModelClass import GPTModel
from gpt_config import GPT_CONFIG_124M


# we tokenize a batch consisting of two text inputs for the GPT model using the tiktoken tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print("batch:\n",batch)

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
# layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# out = layer(batch_example)
# print(out)

""" 
# let’s examine the mean and variance:
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# let’s apply layer normalization to the layer outputs we obtained earlier.
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
# To improve readability, we can also turn off the scientific notation when printing tensor values by setting sci_mode to False:
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)
 """

""" 
# try the LayerNorm module in practice
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n",mean)
print("Variance:\n",var)
 """

# Let’s now initialize the 124-million-parameter GPT model using the GPT_CONFIG_ 124M dictionary
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# Using the numel() method, short for “number of elements,” we can collect the total number of parameters in the model’s parameter tensors:
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# let’s take a look at the shapes of the token embedding layer and linear output layer that we initialized on the model via the GPTModel earlier:
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
# Let’s remove the output layer parameter count from the total GPT-2 model count according to the weight tying:
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(
    f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}"
)

# let’s compute the memory requirements of the 163 million parameters in our GPTModel object
#1 Calculates the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4
#2 Converts to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")