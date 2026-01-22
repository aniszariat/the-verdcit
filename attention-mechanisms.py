import torch

matrix_embedding_layer = [
    [0.43, 0.15, 0.89],  # Your     (x^1)
    [0.55, 0.87, 0.66],  # journey   (x^2)
    [0.57, 0.85, 0.64],  # starts    (x^3)
    [0.22, 0.58, 0.33],  # with      (x^4)
    [0.77, 0.25, 0.10],  # one       (x^5)
    [0.05, 0.80, 0.55],  # step      (x^6)
]


inputs = torch.tensor(matrix_embedding_layer)

"""
query = inputs[1]
# 1:  The second input token serves as the query.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("Attention scores:", attn_scores_2) # ===> tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2) # ===> tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Sum:", attn_weights_2.sum()) # ===> tensor(1.)

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2) # ===> tensor([0.4419, 0.6515, 0.5683])
"""

""" 
print("attention scores using for loops:")
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
print("\n***\n")
print("attention scores using matrix multiplication:")
"""

""" 
1 compute attention socres
2 compute attention weights
3 compute context vecotrs
"""

"""
# In step 1, we add an additional for loop to compute the dot products for all pairs of inputs.
attn_scores = inputs @ inputs.T 
print("1 compute attention socres")
print(attn_scores)

# In step 2, we normalize each row so that the values in each row sum to 1:
attn_weights = torch.softmax(attn_scores, dim=-1)
print("2 compute attention weights")
print(attn_weights)

# In step 3, we use these attention weights to compute all context vectors via matrix multiplication:
all_context_vecs = attn_weights @ inputs
print("3 compute context vecotrs")
print(all_context_vecs)
"""

# Implementing self-attention with trainable weights

#1 The second input element
x_2 = inputs[1]
#2 The input embedding size, d=3
d_in = inputs.shape[1]
#3 The output embedding size, d_out=2
d_out = 2


# we initialize the three weight matrices Wq, Wk, and Wv
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# we compute the query, key, and value vectors:
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)  # ===> tensor([0.4306, 1.4551])