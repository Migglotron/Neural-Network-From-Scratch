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
def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))
    
def sigmoid_derivative(self, x):
    return x * (1 - x)





# Returns vector
def layerSum(activations, weights, biases):
    neuron_output = []
    for i in range(len(weights)):
        for j in range(len(activations)):
            happy = (weights[i][j] * activations[j])# + biases[j]
            neuron_output.append(happy)
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
def random_weight(inputs, neurons):
    weights = [[random.random() for _ in range(inputs) for _ in range(neurons)]]
    return weights

# Randomly initialize the Inputs
def activations(amount):
    return [random.random() for _ in range(amount)]

