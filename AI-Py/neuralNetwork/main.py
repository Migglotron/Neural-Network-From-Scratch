import math
import random
import os
from activationFunction import sigmoid, sigmoid_derivative, ReLU, Softmax, layerSum, outputNeuron, outputLayer, random_weight, activations
from backpropagation import Backpropagation as bp

os.system('clear')


X_train = [[0.0],  # Inputs
            [1.0],
            [1.0]]

Y_train = [[0], [1], [1], [0]] # Biases


Activations =  activations(3) # Inicial input to layer 1, 3 initial neurons

weights = random_weight(3, 5) # Weights for the first layer, 3 inputs, 5 neurons

# outputWeights1 = random_weight(10, 5)  # Weights for the final layer, 10 neurons(output), 10 inputs


            # # Testing # # 
print("Activations:", Activations)
print("Weights:", weights)
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

hidden_layers = layerSum(Activations, weights, 1.0)
final_layer = layerSum(hidden_layers, weights, 1.0)



# # # # Making / activating the layers
# layer1 = layerSum(X_train, weights, biases) # Layer 1
# layer2 = layerSum(layer1, weights, biases) # Layer 2
# outputLayer = layerSum(layer2, outputWeights1, [0.5]) # Output Layer


# # # Print Layer Outputs
print("\nFirst Hidden Layer: ", hidden_layers) # Output Layer
print("\nOutput Layer: ", final_layer) # Output Layer
# # print("\nOutput Layer: ", outputLayer, "\n") # Output Layer

# # cost
# bp.cost(final_layer, X_train[1])

# Re-ajust weights















# # # # Number to Letter Conversion
# alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J","K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U","V", "W", "X", "Y", "Z"]
# letter = []

# for i in range(len(outputLayer)):
#         if outputLayer[i] >= 1:
#                 letter.append(alphabet[i])
#         else:
#                 letter.append("")


# # # #True False
# for i in range(len(outputLayer)):
#         if outputLayer[i] >= 1:
#                 outputLayer[i] = "True"
#         else:
#                 outputLayer[i] = "False"

