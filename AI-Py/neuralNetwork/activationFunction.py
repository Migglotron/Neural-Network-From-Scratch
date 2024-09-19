import math
import random

# ReLU Activation Function
class ReLU():
    def forward(self, x):
        return max(0, x)
    def backward(self, x):
        return 1 if x > 0 else 0

# Softmax Activation functions
class Softmax():
    def forward(self, x):
        exp_values = [math.exp(i) for i in x]  # Exponentiate each element
        sum_exp_values = sum(exp_values)       # Sum of all exponentials
        return [i / sum_exp_values for i in exp_values]  # Normalize
    
    def backward(self, x):
        return [i * (1 - i) for i in x]  # Adjust backward logic for list


# Sigmoid function
class Sigmoid():
    def forward(self, x):
        return 1 / (1 + math.exp(-x))
    def backward(self, x):
        return x * (1 - x)








# Returns vector
def layerSem(activations, weights, baises, function):
    # print(type(a), type(b))
    activationSum = 0
    outputWeights = []
    # Add every weight from every neuron
    for i in range(len(activations)):
        for j in range(len(weights)):
            activationSum += activations[i] * weights[j][i] + baises[j]
        outputWeights.append(activationSum)
        activationSum = 0

    # Makes sure the number of neurons in the layer and the number of biases match
    if len(outputWeights) != len(baises):
        print("Error: The number of neurons in the layer and the number of biases do not match")
        return None

    # Confirms that you put the activation function in to layerSem
    if function == None:
        print("Error: No activation function was given")
        return None
    
    # Different activation functions
    # Apply the activation function
    for i in range(len(outputWeights)):
        if function == "ReLU":
            outputWeights[i] = ReLU.forward(outputWeights[i])
        elif function == "Sigmoid":
            outputWeights[i] = Sigmoid.forward(outputWeights[i])
        elif function == "Softmax":
            outputWeights[i] = Softmax.forward(outputWeights[i])
        # else:
        #     outputWeights[i] = function(outputWeights[i])

    return outputWeights




# Function to handle a single neuron in the output layer
def outputNeuron(activations, weights, bais):
    softmax = Softmax()
    activationSum = 0
    # Add every weight from every neuron
    for i in range(len(activations)):
        activationSum += activations[i] * weights[i]
    
    # Combine the activation sum with the bias
    activationSum += bais
    
    # Returns as a list
    return activationSum

# Function to handle multiple neurons in the output layer and apply Softmax
def outputLayer(activations, weights, biases):
    softmax = Softmax()
    outputSums = []
    
    # Calculate the sum for each neuron in the output layer
    for i in range(len(weights)):
        activationSum = 0
        for j in range(len(activations)):
            activationSum += activations[j] * weights[i][j]
        activationSum += biases
        outputSums.append(activationSum)
    
    # Apply Softmax to the output layer sums (vector)
    return softmax.forward(outputSums)


# Randomly initialize the weights and biases
def initializeWeightsBiases(neurons, inputs): # neurons = number of neurons in the layer, inputs = number of inputs to the layer
    weights = []
    biases = []
    for i in range(neurons): # For each neuron in the layer create a random weight for each input
        neuronWeights = []
        for j in range(inputs):
            neuronWeights.append(random.random())
        weights.append(neuronWeights)
        biases.append(random.random())
    return weights, biases # Return the weights and biases

# Randomly initialize the Inputs
def activations(amount): # Amount = number of numbers that will be in the input
    activations = []
    for i in range(amount): # For each neuron in the layer create a random weight for each input
        randomActivation = []
        for j in range(amount): # For each input to the layer create a random 
            randomActivation.append(random.random())
        activations.append(randomActivation)
    return activations # Return the weights and biases