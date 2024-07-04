import numpy as np
import pandas as pd
import os
import time

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH

# 20 neurons in the first hidden layer
# 15 neurons in the second hidden layer 
HIDDEN_LAYER_1 = 20
HIDDEN_LAYER_2 = 15
OUTPUT_LAYER = 10


def save_parameters(weights, biases, weight_file='weights.npy', bias_file='biases.npy'):
  np.save(f'data/{weight_file}', weights)
  np.save(f'data/{bias_file}', biases)


def load_parameters(weight_file='weights.npy', bias_file='biases.npy'):
  path_weights = f'data/{weight_file}'
  path_biases = f'data/{bias_file}'

  if not os.path.exists(path_weights) or not os.path.exists(path_biases):
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
  else:
    weights = np.load(f'data/{weight_file}', allow_pickle=True).item()
    biases = np.load(f'data/{bias_file}', allow_pickle=True).item()

  return weights, biases


# the derivative of the ReLU function is just
# 1 for positive values and 0 for negative values
def relu_derivative(Z):
  return Z > 0


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


# Function to print training progress
def print_progress(stop_event, progress):
  while not stop_event.is_set():
    print(f'\rEpoch {progress["epoch"]}/{progress["epochs"]}', end="")
    time.sleep(1)
