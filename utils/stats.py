import numpy as np

def print_stats(X: np.ndarray, name_str = ''):
  print(f"{name_str} mean = {round(X.mean(), 2)} std = {round(X.std(), 2)} min = {round(X.min(), 2)} max = {round(X.max(), 2)} X.shape={X.shape},")
