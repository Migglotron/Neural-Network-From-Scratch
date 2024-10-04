import math
import random

# ReLU Activation Function
class ReLU:
    def forward(self, x):
        return max(0, x)

    def backward(self, x):
        return 1 if x > 0 else 0

# Softmax Activation Function
class Softmax:
    def forward(self, x):
        exp_values = [math.exp(i) for i in x]  # Exponentiate each element
        sum_exp_values = sum(exp_values)       # Sum of all exponentials
        return [i / sum_exp_values for i in exp_values]  # Normalize
    
    def backward(self, x):
        # Implementing a proper backward method would require the true labels
        raise NotImplementedError("Backward pass for Softmax not implemented")

# Sigmoid function
class Sigmoid:
    def forward(self, x):
        if isinstance(x, list):
            return [1 / (1 + math.exp(-i)) for i in x]
        else:
            return 1 / (1 + math.exp(-x))
    
    def backward(self, x):
        return x * (1 - x)





# Returns vector
def layerSem(activations, weights, biases):
    neuron_output = []
    # Loop through the weights and activations and get the sum of the products
    for i in range(len(weights)): # For all the group of Weights set
        for j in range(len(activations)): # For all of the Activations from the previous layer
            # print("\nWeight", i, j, ":", weights[i][j]) # for testing purposes
            # print("Activations", i, ":", activations) # For testing purposes
            break

        happy = (weights[i][j] * activations[j]) + biases[j] # Multiply the weights by the activations and adds the bias
        neuron_output.append(happy) # Add the result from "happy" to the neuron_output list

    # # Different activation functions    
    # if function == "Sigmoid":
    #     sigmoid = Sigmoid()
    #     return sigmoid.forward(output_weights)
    # elif function == "ReLU":
    #     relu = ReLU()
    #     return relu.forward(output_weights)
    # elif function == "Softmax":
    #     softmax = Softmax()
    #     return softmax.forward(output_weights)
    # else:
    #     print("Error: Activation function not recognized")
    #     return None

    return neuron_output



# Function to handle a single neuron in the output layer
def outputNeuron(activations, weights, bias):
    activation_sum = sum(activations[i] * weights[i] for i in range(len(activations))) + bias
    return activation_sum

# Function to handle multiple neurons in the output layer and apply Softmax
def outputLayer(activations, weights, biases):
    output_sums = []
    for i in range(len(weights)):
        activation_sum = (activations[j] * weights[i][j] for j in range(len(activations))) + biases[i]
        output_sums.append(activation_sum)
    
    softmax = Softmax()
    return softmax.forward(output_sums)




# Randomly initialize the weights and biases
def initializeWeightsBiases(neurons, inputs):
    weights = [[random.random() for _ in range(inputs)] for _ in range(neurons)]
    biases = [1 for _ in range(neurons)]
    return weights, biases

# Randomly initialize the Inputs
def activations(amount):
    return [random.random() for _ in range(amount)]
