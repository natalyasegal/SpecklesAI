import numpy as np

def concatenate_train_or_val(x_list, y_list):
  x = np.concatenate(x_list, axis=0)
  y = np.concatenate(y_list, axis=0)
  return x, y

def concatenate_test(inp_list):
  return np.concatenate(inp_list, axis=1)
