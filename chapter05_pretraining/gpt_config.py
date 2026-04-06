'''
Considering the GPT_CONFIG_124M dictionary, the only
adjustment we have made compared to the previous chapter
is that we have reduced the context length (context_ length)
to 256 tokens. This modification reduces the computational
demands of training the model, making it possible to carry
out the training on a standard laptop computer.
'''

GPT_CONFIG_124M = {
"vocab_size": 50257,
"context_length": 256,      #1 We shorten the context length from 1,024 to 256 tokens.
"emb_dim": 768,
"n_heads": 12,
"n_layers": 12,
"drop_rate": 0.1,       #2 It’s possible and common to set dropout to 0.
"qkv_bias": False
}