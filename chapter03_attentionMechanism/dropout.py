import torch

# We apply PyTorch’s dropout implementation first to a 6 × 6 tensor consisting of 1s for simplicity:
torch.manual_seed(123)
# We choose a dropout rate of 50%.
dropout = torch.nn.Dropout(0.5)
# 2 Here, we create a matrix of 1s.
example = torch.ones(6, 6)
print(dropout(example))
