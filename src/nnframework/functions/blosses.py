# Implementation of the derivatives of common loss functions

import functions.losses as func

def mean_squared_error(actual, predictions):
    n = len(actual)
    return [-(2 / n) * (actual[i] - predictions[i]) for i in range(n)]

def mean_absolute_error(actual, predictions):
    n = len(actual)
    gradient = []
    for i in range(n):
        diff = predictions[i] - actual[i]
        if diff > 0:
            sign = 1
        elif diff < 0:
            sign = -1
        else:
            sign = 0
        gradient.append(sign / n)
    return gradient

def huber_loss(actual, predictions, delta=1.0):
    n = len(actual)
    gradient = []
    for i in range(n):
        diff = predictions[i] - actual[i]
        abs_diff = abs(diff)

        if abs_diff <= delta:
            gradient.append(diff / n)
        else:
            sign = 1 if diff > 0 else -1
            gradient.append(delta * sign / n)

    return gradient