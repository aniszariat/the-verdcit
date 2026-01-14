# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:45:42 2026

@author: zaria
"""

import re
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
# print(raw_text[:125])

# splits text into individual words and punctuation characters
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# This print statement outputs 4690, which is the number of tokens in this text (without whitespaces)
print("number of tokens:",len(preprocessed))

# print the 1st 30 token from the list
# print(preprocessed[:30])

"""
let’s create a list of all unique tokens and sort them alphabetically
to determine the vocabulary size:
"""
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("vocabulary size:",vocab_size)


# we create the vocabulary and print its first 51 entries
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

tokenizer = tokenizer.SimpleTokenizerV1(vocab)

#text = """"It's the last he painted, you know,"
#Mrs. Gisburn said with pardonable pride."""

#text = "Hello, do you like tea?"

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))