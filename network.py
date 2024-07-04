import numpy as np
import pandas as pd

mnist_train = pd.read_csv('data/mnist_train.csv')

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
# and all the values are probabilites 0-1
# e is used to ensure that even negative values become positive
def softmax(Z):
  # subtracting the max value from Z prevents overflow
  # as we are forcing all the numbers to be negative
  # and a negative exponent makes the value smaller
  expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
  # then divide by the sum of the exponents to normalize the values
  return expZ / expZ.sum(axis=1, keepdims=True)


# forward propagation steps forwards through the network
# X is the input data
def forward_propagation(X, weights, biases):
  # step through the first layer of data
  # this multiplies the input data by the weights and adds the bias
  Z1 = np.dot(X, weights['W1']) + biases['b1']
  # then applies the ReLU activation function
  A1 = relu(Z1)

  # step through the second layer of data
  # this does the same thing but with the values from the previous step
  Z2 = np.dot(A1, weights['W2']) + biases['b2']
  A2 = relu(Z2)

  # step to the output layer
  Z3 = np.dot(A2, weights['W3']) + biases['b3']
  # we use the softmax function instead to get the probabilities
  A3 = softmax(Z3)

  # we store the intermediate values for backpropagation
  cache = (Z1, A1, Z2, A2, Z3, A3)
  # and then return the final output and the cache
  return A3, cache


# the derivative of the ReLU function is just
# 1 for positive values and 0 for negative values
def relu_derivative(Z):
  return Z > 0


def backward_propagation(X, Y, weights, biases, cache, learning_rate):
  # get the intermediate values from the cache
  Z1, A1, Z2, A2, Z3, A3 = cache

  # get the number of samples
  m = X.shape[0]

  # this creates a one-hot encoded matrix for the expected output
  # which is essentially 1 for the correct label and 0 for the rest
  # it uses the numpy eye function to create the identity matrix
  # and pulls the correct row for each label
  y_one_hot = np.eye(OUTPUT_LAYER)[Y]

  # OUTPUT LAYER

  # calculate the gradient for the data
  # or the difference between the actual and expected output
  dZ3 = A3 - y_one_hot
  # here we calculate the gradients for the weights of the output layer
  # divide by the number of samples to get the average gradient
  dW3 = np.dot(A2.T, dZ3) / m
  # calculate the gradients for the biases of the output layer
  # this is just an array of the sum of the gradients
  # on each column, averaged by the sample size
  # it essentially squashes the samples into a single row for each output neuron
  db3 = np.sum(dZ3, axis=0, keepdims=True) / m

  # HIDDEN LAYER 2

  # calculate the gradients for the weights of the second hidden layer
  dA2 = np.dot(dZ3, weights['W3'].T)
  dZ2 = dA2 * relu_derivative(Z2)

  dW2 = np.dot(A1.T, dZ2) / m
  db2 = np.sum(dZ2, axis=0, keepdims=True) / m

  # HIDDEN LAYER 1

  dA1 = np.dot(dZ2, weights['W2'].T)
  dZ1 = dA1 * relu_derivative(Z1)

  dW1 = np.dot(X.T, dZ1) / m
  db1 = np.sum(dZ1, axis=0, keepdims=True) / m

  # update the weights and biases
  weights['W1'] -= learning_rate * dW1
  biases['b1'] -= learning_rate * db1

  weights['W2'] -= learning_rate * dW2
  biases['b2'] -= learning_rate * db2

  weights['W3'] -= learning_rate * dW3
  biases['b3'] -= learning_rate * db3

