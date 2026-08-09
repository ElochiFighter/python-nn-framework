# Implementation of the derivatives of common activation functions

import math
import functions.activations as func

def ReLU(x):
    return 1.0 if x > 0 else 0.0

def Sigmoid(x):
    s = func.Sigmoid(x)
    return s * (1 - s)

def TanH(x):
    t = func.TanH(x)
    return 1 - t ** 2

def Softmax(vector):
    pass  # Softmax derivative is more complex and usually handled differently in practice

