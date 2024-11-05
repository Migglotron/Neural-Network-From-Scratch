import math
import random
from activationFunction import sigmoid, sigmoid_derivative, ReLU, Softmax, layerSum, outputNeuron, random_weight, activations
from backpropagation import Backpropagation


training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

bp = Backpropagation(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
epochs = 5000
bp.train(training_data, epochs)

# Activations =  activations(2) # Inicial input to layer 1

# weights, biases = initializeWeightsBiases(3, 2) # Mkaes weights and biases for layers, 3 neurons(output), 2 inputs

# outputWeights1, gliases = initializeWeightsBiases(2, 3)  # Weights for the final layer, 2 neurons(output), 3 inputs


            # # Testing # # 
print("Activations:", Activations)
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


hidden_input = [[layerSum(X_train[i], [bp.weights_input_hidden[j][k] for j in range(bp.input_size)]) + bp.bias_hidden[k] for k in range(bp.hidden_size)] for i in range(len(X_train))]
hidden_output = [[sigmoid(hidden_input[i][j]) for j in range(bp.hidden_size)] for i in range(len(hidden_input))]
final_input = [[layerSum(hidden_output[i], [bp.weights_hidden_output[j][k] for j in range(bp.hidden_size)]) + bp.bias_output[k] for k in range(bp.output_size)] for i in range(len(hidden_output))]
final_output = [[sigmoid(final_input[i][j]) for j in range(bp.output_size)] for i in range(len(final_input))]


# # # # Making / activating the layers
# layer1 = layerSum(Activations, weights, biases) # Layer 1
# layer2 = layerSum(layer1, weights, biases) # Layer 2
# outputLayer = layerSum(layer2, outputWeights1, gliases) # Output Layer


# # # Print Layer Outputs
print("\nFirst Hidden Layer: ", hidden_output) # Output Layer
# print("\nSecond Hidden Layer: ", layer2) # Output Layer
print("\nOutput Layer: ", final_output, "\n") # Output Layer

# Backward pass
for i in range(len(X_train)):
    bp.backward_pass(X_train[i], Y_train[i], hidden_output[i], final_output[i])