# backpropagation.py

import math
import random
from activationFunction import sigmoid, sigmoid_derivative, random_weight

class Backpropagation:
    def cost(outputs, expected):
        costs = []
        for i in range(len(outputs)):
            dif = (outputs[i] - expected[i])
            costs.append(dif)
    def agustment():
        return