import numpy as np
import pandas as pd
import os


class Network:
  def __init__(self, input_size=784, output_size=10, **kwargs):
    self.input_size = input_size
    self.output_layer = output_size

    self.hidden_layer_1 = kwargs.get('hidden_layer_1', 20)
    self.hidden_layer_2 = kwargs.get('hidden_layer_2', 20)

