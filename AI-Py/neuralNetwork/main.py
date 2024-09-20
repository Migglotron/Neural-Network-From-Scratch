import math
import random
from activationFunction import Sigmoid, ReLU, Softmax, activations, layerSem, outputLayer, initializeWeightsBiases, outputNeuron
from backpropagation import Backpropagation, crossEntropyLoss


Activations =  activations(2) # Inicial input to layer 1

weights, biases = initializeWeightsBiases(3, 2)  # Weights for the hidden layers

outputWeights1 = initializeWeightsBiases(2, 3)  # Weights for the final layer


# print(type(Activations))
# print(type(weights))
# print(type(biases))



# Making / activating the layers
layer1 = layerSem(Activations, weights, biases, "Sigmoid") # Layer 1
layer2 = layerSem(layer1, weights, biases, "Sigmoid") # Layer 2
outputLayer = outputLayer(layer2, outputWeights1, 0) # Output Layer


# # Print Layer Outputs
# print("Output Layer: ", outputLayer) # Output Layer