import math

# ReLU Activation Function
class ReLU():
    def forward(self, x):
        return max(0, x)
    def backward(self, x):
        return 1 if x > 0 else 0

# Softmax Activation functions
class Softmax():
    def forward(self, x):
        return math.exp(x) / sum(math.exp(i) for i in x)
    def backward(self, x):
        return x * (1 - x)

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

    for i in range(len(outputWeights)):
        outputWeights[i] = function(outputWeights[i])

    return outputWeights
    