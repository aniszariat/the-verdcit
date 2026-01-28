#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:47:10 2026

@author: dell
"""

from importlib.metadata import version
print("tiktoken version:", version("tiktoken"))
import tiktoken
print(dir(tiktoken))
print(tiktoken.__file__)
