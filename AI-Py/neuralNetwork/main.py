import math
import random
from activationFunction import sigmoid, sigmoid_derivative, ReLU, Softmax, layerSum, outputNeuron, outputLayer, random_weight
from backpropagation import Backpropagation

X_train = [[0, 0, 1],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 1]]

y_train = [[0], [1], [1], [0]]

# Inisializing the Backpropagation
bp = Backpropagation(5, 4, 1, 0.01) # 5 inputs, 10 hidden neurons, 5 output neurons, learning rate 0.1


Activations =  activations(5) # Inicial input to layer 1

weights = random_weight(5, 10) # Weights for the first layer, 5 inputs, 10 neurons

outputWeights1 = random_weight(10, 5)  # Weights for the final layer, 10 neurons(output), 10 inputs


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



# # # # Making / activating the layers
# layer1 = layerSum(X_train, weights, biases) # Layer 1
# layer2 = layerSum(layer1, weights, biases) # Layer 2
# outputLayer = layerSum(layer2, outputWeights1, [0.5]) # Output Layer

hidden_input = [[bp.dot_product(X_train[i], [bp.weights_input_hidden[j][k] for j in range(bp.input_size)]) + bp.bias_hidden[k] for k in range(bp.hidden_size)] for i in range(len(X_train))]
hidden_output = [[bp.sigmoid(hidden_input[i][j]) for j in range(bp.hidden_size)] for i in range(len(hidden_input))]
final_input = [[bp.dot_product(hidden_output[i], [bp.weights_hidden_output[j][k] for j in range(bp.hidden_size)]) + bp.bias_output[k] for k in range(bp.output_size)] for i in range(len(hidden_output))]
final_output = [[bp.sigmoid(final_input[i][j]) for j in range(bp.output_size)] for i in range(len(final_input))]



# # # Print Layer Outputs
print("\nFirst Hidden Layer: ", hidden_output) # Output Layer
print("\nSecond Hidden Layer: ", final_output) # Output Layer
# print("\nOutput Layer: ", outputLayer, "\n") # Output Layer



# Backward pass
for i in range(len(X_train)):
    bp.backward_pass(X_train[i], y_train[i], hidden_output[i], final_output[i])
















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

