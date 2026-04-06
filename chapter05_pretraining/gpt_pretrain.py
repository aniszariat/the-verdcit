import torch
from chapter04_GPTModel.GPTModelClass import GPTModel
from gpt_config import GPT_CONFIG_124M

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()