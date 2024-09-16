import math
from activationFunction import Sigmoid, ReLU, Softmax, layerSem


Activations = [1.4, -0.1, 0.5]  # input

Weights = [[-0.4, 0.3, 1.0],
            [-0.2, 0.6, -1],
            [-0.4, 0.2, 1]]  # weights

biases = [0.1, 0.2, 0.3]  # biases
# Making a layer with math

layer1 = layerSem(Activations, Weights, biases, math.tanh)  # layer1
layer2 = layerSem(layer1, Weights, biases, math.tanh)  # layer2

for i in layer2:
    print(i)