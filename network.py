import numpy as np
import os
import pandas as pd
from utils import *
import threading

mnist_train = pd.read_csv('data/mnist_train.csv')

# x are the specific pixels of the images
# this takes all the rows and all the columns except for the first one
# each row is an image and each column within that row is a pixel
X_train = mnist_train.iloc[:, 1:].values

# y are the labels or expected output of the images.
# this takes all the rows of just the first column
# each row is an image and the value saved here is the label for that image
Y_train = mnist_train.iloc[:, 0].values

weights, biases = load_parameters()

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


def train(X, Y, weights, biases, epochs, batch_size, learning_rate):
  print("starting training")

  m = X.shape[0]
  progress = {"epoch": 0, "epochs": epochs}
  stop_event = threading.Event()

  # Start progress printing thread
  progress_thread = threading.Thread(target=print_progress, args=(stop_event, progress))
  progress_thread.start()


  for epoch in range(epochs):
    # shuffle the data
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    for i in range(0, m, batch_size):
      X_batch = X_shuffled[i:i+batch_size]
      Y_batch = Y_shuffled[i:i+batch_size]

      # forward propagation
      A3, cache = forward_propagation(X_batch, weights, biases)

      # backward propagation
      backward_propagation(X_batch, Y_batch, weights, biases, cache, learning_rate)

    # Update epoch in progress
    progress["epoch"] = epoch + 1

  # Stop progress thread
  stop_event.set()
  progress_thread.join()

  print(f'\rEpoch {progress["epoch"]}/{progress["epochs"]}')
  
  A3, _ = forward_propagation(X, weights, biases)
  loss = -np.mean(np.log(A3[np.arange(len(Y)), Y]))
  accuracy = np.mean(np.argmax(A3, axis=1) == Y)
  print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

  # save the data
  save_parameters(weights, biases)