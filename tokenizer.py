# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 21:30:36 2026

@author: zaria
"""

import re
class SimpleTokenizerV1:
    def __init__(self, vocab):
        #1 Stores the vocabulary as a class attribute for access in the encode and decode methods
        self.str_to_int = vocab #1
        #2 Creates an inverse vocabulary that maps token IDs back to the original text tokens
        self.int_to_str = {i:s for s,i in vocab.items()} #2
    
    #3 Processes input text into token IDs
    def encode(self, text): #3
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()]
        
        
        #6 Replaces unknown words by <|unk|> tokens
        preprocessed = [item if item in self.str_to_int #6
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    #4 Converts token IDs back into text
    def decode(self, ids): #4
        text = " ".join([self.int_to_str[i] for i in ids])
        #5 Removes spaces before the specified punctuation
        #text = re.sub(r'\s+([,.?!"()\'])', r'\\1', text) #5
        #7 Replaces spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\\1', text) #7
        return text
