# Loss function for neural network
# Cost function: Cross-entropy loss function
# Find the difference between the predicted value and the actual value
# Calculate the best step to take to find the "valley" of the cost function
#
# Backpropagation: The process of finding the derivative of the cost function
#  with respect to the weights and biases
#


import math

# Cross-entropy loss function
def crossEntropyLoss(predicted, actual):
    return -actual * math.log(predicted) - (1 - actual) * math.log(1 - predicted) # Find the difference between the predicted value and the actual value

# Backpropagation
class Backpropagation():
    def __init__(self, learningRate):   # learningRate is the step size that we take to find the "valley"
        self.learningRate = learningRate 
        self.error = 0 # The error of the neural network (cost function)
    def forward(self, predicted, actual):
        self.error = crossEntropyLoss(predicted, actual)    # Find the error of the neural network (cost function)
        return self.error   # Return the error of the neural network (cost function)
    def backward(self, predicted, actual):
        return (predicted - actual) * self.learningRate     # Find the derivative of the cost function with respect to the weights and biases


# Test the backpropagation
backprop = Backpropagation(0.1)     # Create a backpropagation object with a learning rate of 0.1
print(backprop.forward(0.8, 1))     # Find the error of the neural network (cost function)
print(backprop.backward(0.8, 1))    # Find the derivative of the cost function with respect to the weights and biases



# Summary:
# Adds the backpropagation backward to the weights and biases of the neural network
#  to find the best step to take to find the "valley" of the cost function
#  and minimize the error of the neural network (cost function)
#
# The backpropagation algorithm is used to train the neural network
