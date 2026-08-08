# Implementation of common loss functions

import math

def ReLU(x):
    return max(0, x)

def LeakyReLU(x, alpha=0.01):
    return x if x > 0 else alpha * x

def Sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def TanH(x):
    return math.tanh(x)

def Softmax(vector):
    max_val = max(vector)
    exp_vector = [math.exp(x - max_val) for x in vector]
    sum_exp = sum(exp_vector)
    return [exp_x / sum_exp for exp_x in exp_vector]


def Swish(x, beta=1.0):
    clamped_bx = max(-500, min(500, beta * x))
    sigmoid_bx = 1 / (1 + math.exp(-clamped_bx))
    return x * sigmoid_bx

def GELU(x):
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * (x ** 3))
    return 0.5 * x * (1 + math.tanh(inner))

def ELU(x, alpha=1.0):
    return x if x > 0 else alpha * math.expm1(x)

def SELU(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * (x if x > 0 else alpha * math.expm1(x))

def Mish(x):
    clamped_x = max(-500, min(500, x))
    softplus = math.log1p(math.exp(clamped_x))
    return x * math.tanh(softplus)
