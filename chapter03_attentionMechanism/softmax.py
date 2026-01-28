#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:34:29 2026

@author: dell
"""

"""
import numpy as np

def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

values = [0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656]
print(softmax(values))
"""
import math

def softmax(x):
    max_x = max(x)  # for numerical stability
    exp_values = [math.exp(i - max_x) for i in x]
    sum_exp = sum(exp_values)
    return [v / sum_exp for v in exp_values]


values = [0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656]
print(softmax(values))
