import numpy as np
import pandas as pd
import os
import time


# using the ReLU activation function
# which is just sets all negative values to 0
def relu(Z):
  return np.maximum(0, Z)


# the derivative of the ReLU function is just
# 1 for positive values and 0 for negative values
def relu_derivative(Z):
  return Z > 0


def sigmoid(Z):
  ...


def sigmoid_derivative(Z):
  ...


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
