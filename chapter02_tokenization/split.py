# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:53:24 2026

@author: zaria
"""

import re
text = "Hello, world. This, is a test."
# split a text on whitespace characters
result = re.split(r'(\s)', text)
# output ===> ['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']

# split on whitespaces (\s), commas, and periods ([,.])
result = re.split(r'([,.]|\s)', text)
# output ===> ['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']

# remove these redundant characters
result = [item for item in result if item.strip()]
# output ===> ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
# output ===> ['Hello', ',', 'world', '.', 'Is this', '--', 'a test', '?']