# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:45:42 2026

@author: zaria
"""

import re
import tiktoken

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
# print(raw_text[:125])

"""
splits text into individual words and punctuation characters
""" 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# This print statement outputs 4690, which is the number of tokens in this text (without whitespaces)
print("number of tokens:",len(preprocessed))

# print the 1st 30 token from the list
# print(preprocessed[:30])

"""
we add an <|unk|> token to represent new and unknown words that were not part of the training data and thus not part of the existing vocabulary. 
we add an <|endoftext|> token that we can use to separate two unrelated text sources.
"""
preprocessed.extend(["<|endoftext|>", "<|unk|>"])
"""
let’s create a list of all unique tokens and sort them alphabetically
to determine the vocabulary size:
"""
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("vocabulary size:",vocab_size)


# we create the vocabulary and print its first 51 entries
vocab = {token:integer for integer,token in enumerate(all_words)}

text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
)
"""
instantiate the BPE tokenizer fromtiktoken as follows:
"""
tokenizer = tiktoken.get_encoding("gpt2")
print(text)
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)
print(tokenizer.decode(ids))
