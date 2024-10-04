import math
import random
from activationFunction import Sigmoid, ReLU, Softmax, activations, layerSem, outputLayer, initializeWeightsBiases, outputNeuron
from backpropagation import Backpropagation, crossEntropyLoss


Activations =  activations(2) # Inicial input to layer 1

weights, biases = initializeWeightsBiases(3, 2) # Mkaes weights and biases for layers, 3 neurons(output), 2 inputs

outputWeights1, gliases = initializeWeightsBiases(2, 3)  # Weights for the final layer, 2 neurons(output), 3 inputs


            # # Testing # #  
# print("Activations:", Activations)
# print("Weights:", weights)
# print("Biases:", biases)
# print("Output Weights:", outputWeights1)

# hell = []
# for i in range(len(weights)):
#     for j in range(len(Activations)):
#         print("\nWeight", i, j, ":", weights[i][j])
#         print("Activations", i, ":", Activations)
    
#     happy = (weights[i][j] * Activations[j])
#     hell.append(happy)

# print("Expected Output:", hell)
        # # End of Testing # #



# # # Making / activating the layers
layer1 = layerSem(Activations, weights, biases) # Layer 1
layer2 = layerSem(layer1, weights, biases) # Layer 2
outputLayer = layerSem(layer2, outputWeights1, gliases) # Output Layer


# # # Print Layer Outputs
print("\nFirst Hidden Layer: ", layer1) # Output Layer
print("\nSecond Hidden Layer: ", layer2) # Output Layer
print("\nOutput Layer: ", outputLayer, "\n") # Output Layer