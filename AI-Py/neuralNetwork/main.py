import math
import random
from activationFunction import Sigmoid, ReLU, Softmax, layerSem, outputNeuron, outputLayer, initializeWeightsBiases, activations


Activations =  activations(2) # Inicial input to layer 1

weights, biases = initializeWeightsBiases(3, 2)  # Weights for the hidden layers

outputWeightsNeuron = [[-0.1, 0.3, 1.1], [-0.7, 0.9, 1.3]]  # Weights for the final layer


print(Activations)
print(weights)
print(biases)



# Making / activating the layers
# layer1 = layerSem(Activations, weights, biases, Sigmoid)  # layer1
# layer2 = layerSem(layer1, weights, biases, Sigmoid)  # layer2

# finalOutput = outputLayer(layer2, outputWeightsNeuron, 0.1)  # Final Output


# Print Layer Outputs
# print("Layer 1 :", layer1)  # layer1
# print("Layer 2 :", layer2)  # layer2
# print("Final Outputs :", finalOutput)  # Final Output