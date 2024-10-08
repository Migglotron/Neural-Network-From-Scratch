# # Self made test data for neural network
# import random

# def dataset():
#     # Random dataset
#     dataset = []
#     for i in range(5):
#         dataset.append(random.randint(0, 1))
#     return dataset

# print(dataset()) # Random dataset




# # # # MNIST DATASET # # # #

import mnist

# Load the MNIST dataset
train_images = mnist.train_images() # Training images
train_labels = mnist.train_labels() # Training labels

test_images = mnist.test_images() # Testing images
test_labels = mnist.test_labels() # Testing labels

# Print the shape of the training images
print("Training images shape:", train_images.shape)

