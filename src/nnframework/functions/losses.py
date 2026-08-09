# Implementation of common activation functions

import math
from collections.abc import Sequence
from numbers import Number

def mean_squared_error(actual: Sequence[Number], predictions: Sequence[Number]):
    squared_errors = []
    for i in range(len(actual)):
        diff = actual[i] - predictions[i]
        squared_errors.append(diff ** 2)
        
    return sum(squared_errors) / len(squared_errors)     

def mean_absolute_error(actual: Sequence[Number], predictions: Sequence[Number]):
    errors = []
    for i in range(len(actual)):
        diff = actual[i] - predictions[i]
        errors.append(abs(diff))
        
    return sum(errors) / len(errors)  

def huber_loss(actual: Sequence[Number], predictions: Sequence[Number], delta: Number = 1.0):
    huber_errors = []
    for i in range(len(actual)):
        diff = actual[i] - predictions[i]
        abs_diff = abs(diff)
        
        if abs_diff <= delta:
            huber_errors.append(0.5 * (diff ** 2))
        else:
            huber_errors.append(delta * (abs_diff - 0.5 * delta))
            
    return sum(huber_errors) / len(huber_errors)
        


