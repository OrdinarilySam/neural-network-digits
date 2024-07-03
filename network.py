import numpy as np
import pandas as pd

mnist_train = pd.read_csv('mnist_train.csv')

# x are the specific pixels of the images
# this takes all the rows and all the columns except for the first one
# each row is an image and each column within that row is a pixel
X_train = mnist_train.iloc[:, 1:].values

# y are the labels or expected output of the images.
# this takes all the rows of just the first column
# each row is an image and the value saved here is the label for that image
Y_train = mnist_train.iloc[:, 0].values

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH

# 20 neurons in the first hidden layer
# 15 neurons in the second hidden layer 
HIDDEN_LAYER_1 = 20
HIDDEN_LAYER_2 = 15
OUTPUT_LAYER = 10

# this creates matrices of random weights for each layer
# the matrices are L1 x L2 where L1 is the number of neurons in the previous layer
# scaled by 0.01 from a normal distribution
weights = {
  "W1": np.random.randn(INPUT_SIZE, HIDDEN_LAYER_1) * 0.01,
  "W2": np.random.randn(HIDDEN_LAYER_1, HIDDEN_LAYER_2) * 0.01,
  "W3": np.random.randn(HIDDEN_LAYER_2, OUTPUT_LAYER) * 0.01
}

# this creates arrays (or 1d matrices) of zeros for each layer
biases = {
  "b1": np.zeros((1, HIDDEN_LAYER_1)),
  "b2": np.zeros((1, HIDDEN_LAYER_2)),
  "b3": np.zeros((1, OUTPUT_LAYER))
}

# using the ReLU activation function
# which is just sets all negative values to 0
def relu(Z):
  return np.maximum(0, Z)

# the softmax activation function
# it scales the output so that the sum of the output is 1
def softmax(Z):
  expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
  return expZ / expZ.sum(axis=1, keepdims=True)
