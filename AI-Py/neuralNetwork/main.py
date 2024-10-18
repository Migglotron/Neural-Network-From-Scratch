import math
import random
from activationFunction import Sigmoid, ReLU, Softmax, activations, layerSum, outputLayer, initializeWeightsBiases, outputNeuron
from backpropagation import Backpropagation, crossEntropyLoss


Activations =  activations(5) # Inicial input to layer 1

weights, biases = initializeWeightsBiases(10, 5) # Mkaes weights and biases for layers, 10 neurons(output), 5 inputs

outputWeights1, gliases = initializeWeightsBiases(25, 10)  # Weights for the final layer, 26 neurons(output), 10 inputs


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
layer1 = layerSum(Activations, weights, biases) # Layer 1
layer2 = layerSum(layer1, weights, biases) # Layer 2
outputLayer = layerSum(layer2, outputWeights1, [0.5]) # Output Layer


# # # Print Layer Outputs
print("\nFirst Hidden Layer: ", layer1) # Output Layer
print("\nSecond Hidden Layer: ", layer2) # Output Layer
print("\nOutput Layer: ", outputLayer, "\n") # Output Layer


# # # Number to Letter Conversion
alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J","K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U","V", "W", "X", "Y", "Z"]
letter = []

for i in range(len(outputLayer)):
        if outputLayer[i] >= 1:
                letter.append(alphabet[i])
        else:
                letter.append("")


# # # #True False
# for i in range(len(outputLayer)):
#         if outputLayer[i] >= 1:
#                 outputLayer[i] = "True"
#         else:
#                 outputLayer[i] = "False"


print("Letter Output Layer: ", letter)
# print ("True/False Output Layer: ", outputLayer)
