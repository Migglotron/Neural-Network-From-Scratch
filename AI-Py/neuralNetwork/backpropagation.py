import json
import random
import math

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize parameters
        self.learning_rate = learning_rate

        # Weights and biases
        # Input to hidden layer weights and biases
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        
        # Hidden to output layer weights and biases
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        sx = self.sigmoid(x)
        return sx * (1 - sx)

    def forward(self, inputs):
        hidden_activations = [self.sigmoid(sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(len(inputs))) + self.bias_hidden[j])
                              for j in range(len(self.weights_input_hidden[0]))]
        outputs = [self.sigmoid(sum(hidden_activations[j] * self.weights_hidden_output[j][k] for j in range(len(hidden_activations))) + self.bias_output[k])
                   for k in range(len(self.weights_hidden_output[0]))]
        return hidden_activations, outputs

    def backpropagation(self, inputs, hidden_activations, outputs, expected_outputs):
        output_deltas = [(outputs[k] - expected_outputs[k]) * outputs[k] * (1 - outputs[k]) for k in range(len(outputs))]
        hidden_deltas = [sum(self.weights_hidden_output[j][k] * output_deltas[k] for k in range(len(output_deltas))) * hidden_activations[j] * (1 - hidden_activations[j])
                         for j in range(len(hidden_activations))]

        for j in range(len(self.weights_hidden_output)):
            for k in range(len(self.weights_hidden_output[j])):
                self.weights_hidden_output[j][k] -= self.learning_rate * output_deltas[k] * hidden_activations[j]
        for k in range(len(self.bias_output)):
            self.bias_output[k] -= self.learning_rate * output_deltas[k]

        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[i])):
                self.weights_input_hidden[i][j] -= self.learning_rate * hidden_deltas[j] * inputs[i]
        for j in range(len(self.bias_hidden)):
            self.bias_hidden[j] -= self.learning_rate * hidden_deltas[j]

    def train(self, training_data, epochs):
        for epoch in range(epochs):
            for inputs, expected_outputs in training_data:
                hidden_activations, outputs = self.forward(inputs)
                self.backpropagation(inputs, hidden_activations, outputs, expected_outputs)

    def save_parameters(self, filename="network_parameters.json"):
        params = {
            "weights_input_hidden": self.weights_input_hidden,
            "bias_hidden": self.bias_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_output": self.bias_output
        }
        with open(filename, 'w') as f:
            json.dump(params, f)
    
    def load_parameters(self, filename="network_parameters.json"):
        try:
            with open(filename, 'r') as f:
                params = json.load(f)
            self.weights_input_hidden = params["weights_input_hidden"]
            self.bias_hidden = params["bias_hidden"]
            self.weights_hidden_output = params["weights_hidden_output"]
            self.bias_output = params["bias_output"]
            print("Parameters loaded successfully.")
        except FileNotFoundError:
            print("No saved parameters found; starting with random initialization.")

# Demonstration of saving and loading functionality
# Instantiate and train the network, then save its state
nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)

# Train on XOR data
training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [0]), ([1, 1], [1])]
epochs = 5000
nn.train(training_data, epochs)

# Save the trained parameters
nn.save_parameters()

# Load parameters in a new instance to continue training or predict
nn2 = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
nn2.load_parameters()  # This will load the saved weights and biases

# Test to verify outputs from the loaded network are consistent
results = []
for inputs, expected_output in training_data:
    _, output = nn2.forward(inputs)
    results.append((inputs, output, expected_output))
    print("Outputs:", output)

print(results)

