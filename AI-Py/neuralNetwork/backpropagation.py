# backpropagation.py

import math
import random
from activationFunction import sigmoid, sigmoid_derivative, random_weight

class Backpropagation:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = [[random_weight() for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[random_weight() for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_hidden = [0.0 for _ in range(hidden_size)]
        self.bias_output = [0.0 for _ in range(output_size)]
    
    def backward_pass(self, X, y, hidden_output, final_output):
        # Calculate the error
        error = [y[i] - final_output[i] for i in range(len(y))]
        d_output = [error[i] * self.sigmoid_derivative(final_output[i]) for i in range(len(error))]
        
        # Error for hidden layer
        error_hidden_layer = [self.dot_product(d_output, [self.weights_hidden_output[j][i] for j in range(len(d_output))]) for i in range(self.hidden_size)]
        d_hidden_layer = [error_hidden_layer[i] * self.sigmoid_derivative(hidden_output[i]) for i in range(self.hidden_size)]
        
        # Update weights and biases
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] += hidden_output[i] * d_output[j] * self.learning_rate
            self.bias_output[i] += d_output[i] * self.learning_rate
        
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += X[i] * d_hidden_layer[j] * self.learning_rate
            self.bias_hidden[i] += d_hidden_layer[i] * self.learning_rate