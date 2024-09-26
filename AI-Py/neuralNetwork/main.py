import math
import random
from activationFunction import Sigmoid, ReLU, Softmax, activations, layerSem, outputLayer, initializeWeightsBiases, outputNeuron
from backpropagation import Backpropagation, crossEntropyLoss


Activations =  activations(2) # Inicial input to layer 1

weights, biases = initializeWeightsBiases(3, 2)  # Weights for the hidden layers

outputWeights1 = initializeWeightsBiases(2, 3)  # Weights for the final layer




# # Testing # #

# output_weights = []

# print(Activations)
# print(weights)

# for i in range(len(Activations)):
#     for j in range(len(weights[i])):
#         activation_sum = Activations[0] * weights[0][0] + biases[0]
#         activation_sum1 = Activations[1] * weights[0][1] + biases[1]

# print("Output" ,activation_sum + activation_sum1)

# # End of testing # #



# # # Making / activating the layers
layer1 = layerSem(Activations, weights, biases, "Sigmoid") # Layer 1
# layer2 = layerSem(layer1, weights, biases, "Sigmoid") # Layer 2
# outputLayer = outputLayer(layer2, outputWeights1, 0) # Output Layer


# # # Print Layer Outputs
print("Output Layer: ", layer1) # Output Layer