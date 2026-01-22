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
