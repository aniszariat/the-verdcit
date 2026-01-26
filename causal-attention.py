import torch
from selfAttentionClass import SelfAttention_v2

torch.manual_seed(789)
matrix_embedding_layer = [
    [0.43, 0.15, 0.89],  # Your     (x^1)
    [0.55, 0.87, 0.66],  # journey   (x^2)
    [0.57, 0.85, 0.64],  # starts    (x^3)
    [0.22, 0.58, 0.33],  # with      (x^4)
    [0.77, 0.25, 0.10],  # one       (x^5)
    [0.05, 0.80, 0.55],  # step      (x^6)
]
inputs = torch.tensor(matrix_embedding_layer)
d_in = inputs.shape[1]  # The input embedding size, d=3
d_out = 2  # The output embedding size, d_out=2
sa_v2 = SelfAttention_v2(d_in, d_out)
# Reuses the query and key weight matrices of the SelfAttention_v2 object from the previous section for convenience
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
# attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# print("attention weights 1:")
# print(attn_weights)

# # # Applying a causal attention mask
"""
The softmax function converts its inputs into a probability
distribution. When negative infinity values (-∞) are present
in a row, the softmax function treats them as zero
probability.
(Mathematically, this is because e –∞ approaches 0.)
"""
context_length = attn_scores.shape[0]
# We can implement this more efficient masking “trick”
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print("masked:")
# print(masked)

# the modified attention weights
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print("the modified attention weights:")
print(attn_weights)

# # # Masking additional attention weights with dropout

torch.manual_seed(123)
# We choose a dropout rate of 50%.
dropout = torch.nn.Dropout(0.5)
print("dropout:")
print(dropout(attn_weights))
