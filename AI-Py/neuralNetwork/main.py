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
letter = []
for i in range(len(outputLayer)):
        if outputLayer[i] >= 1.0:
                if i == 0:
                        outputLayer[i] = "A"
                        letter.append(outputLayer[i])
                elif i == 1:
                        outputLayer[i] = "B"
                        letter.append(outputLayer[i])
                elif i == 2:
                        outputLayer[i] = "C"
                        letter.append(outputLayer[i])
                elif i == 3:
                        outputLayer[i] = "D"
                        letter.append(outputLayer[i])
                elif i == 4:
                        outputLayer[i] = "E"
                        letter.append(outputLayer[i])
                elif i == 5:
                        outputLayer[i] = "F"
                        letter.append(outputLayer[i])
                elif i == 6:
                        outputLayer[i] = "G"
                        letter.append(outputLayer[i])
                elif i == 7:
                        outputLayer[i] = "H"
                        letter.append(outputLayer[i])
                elif i == 8:
                        outputLayer[i] = "I"
                        letter.append(outputLayer[i])
                elif i == 9:
                        outputLayer[i] = "J"
                        letter.append(outputLayer[i])
                elif i == 10:
                        outputLayer[i] = "K"
                        letter.append(outputLayer[i])
                elif i == 11:
                        outputLayer[i] = "L"
                        letter.append(outputLayer[i])
                elif i == 12:
                        outputLayer[i] = "M"
                        letter.append(outputLayer[i])
                elif i == 13:
                        outputLayer[i] = "N"
                        letter.append(outputLayer[i])
                elif i == 14:
                        outputLayer[i] = "O"
                        letter.append(outputLayer[i])
                elif i == 15:
                        outputLayer[i] = "P"
                        letter.append(outputLayer[i])
                elif i == 16:
                        outputLayer[i] = "Q"
                        letter.append(outputLayer[i])
                elif i == 17:
                        outputLayer[i] = "R"
                        letter.append(outputLayer[i])
                elif i == 18:
                        outputLayer[i] = "S"
                        letter.append(outputLayer[i])
                elif i == 19:
                        outputLayer[i] = "T"
                        letter.append(outputLayer[i])
                elif i == 20:
                        outputLayer[i] = "U"
                        letter.append(outputLayer[i])
                elif i == 21:
                        outputLayer[i] = "V"
                        letter.append(outputLayer[i])
                elif i == 22:
                        outputLayer[i] = "W"
                        letter.append(outputLayer[i])
                elif i == 23:
                        outputLayer[i] = "X"
                        letter.append(outputLayer[i])
                elif i == 24:
                        outputLayer[i] = "Y"
                        letter.append(outputLayer[i])
                elif i == 25:
                        outputLayer[i] = "Z"
                        letter.append(outputLayer[i])


# # # #True False
# for i in range(len(outputLayer)):
#         if outputLayer[i] >= 1:
#                 outputLayer[i] = "True"
#         else:
#                 outputLayer[i] = "False"


print("Letter Output Layer: ", letter)
# print ("True/False Output Layer: ", outputLayer)
