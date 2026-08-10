# Implementation of the derivatives of common activation functions

import math
from . import activations as func

def ReLU(x):
    return 1.0 if x > 0 else 0.0

def Sigmoid(x):
    s = func.Sigmoid(x)
    return s * (1 - s)

def TanH(x):
    t = func.TanH(x)
    return 1 - t ** 2

def Softmax(vector):
    n = len(vector)
    jacobian = []
    for k in range(n):
        row = []
        for i in range(n):
            indicator = 1.0 if k == i else 0.0
            row.append(vector[k] * (indicator - vector[i]))
        jacobian.append(row)
    return jacobian

def Swish(x, beta=1.0):
    s = func.Swish(x, beta)
    sigmoid_bx = 1 / (1 + math.exp(-beta * x))
    return sigmoid_bx + beta * sigmoid_bx * (x - s)

def GELU(x):
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * (x ** 3))
    tanh_inner = math.tanh(inner)
    return 0.5 * (1 + tanh_inner) + (0.5 * x * (1 - tanh_inner ** 2) * (math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * (x ** 2))))

def ELU(x, alpha=1.0):
    return 1.0 if x > 0 else alpha * math.exp(x)

def SELU(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale if x > 0 else scale * alpha * math.exp(x)

def Mish(x):
    clamped_x = max(-500, min(500, x))
    softplus = math.log1p(math.exp(clamped_x))
    tanh_softplus = math.tanh(softplus)
    return tanh_softplus + clamped_x * (1 - tanh_softplus ** 2) * (1 / (1 + math.exp(-clamped_x)))