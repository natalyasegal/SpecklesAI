import numpy as np

def save_dataset_x(x, file_name):
  print(f'Saving {file_name}')
  with open(file_name, 'wb') as f:
      np.save(f, x)

def save_dataset(x, y, file_name):
  print(f'Saving {file_name}')
  with open(file_name, 'wb') as f:
      np.save(f, x)
      np.save(f, y)

def load_dataset(file_name):
  print(f'Loading {file_name}')
  try:
    with open(file_name, 'rb') as f:
      x = np.load(f)
      y = np.load(f)
  except EOFError:
    print(f"Failed to load data from {f}, file may be corrupted or empty.")
    assert(False)
  return x, y

def load_dataset_x(file_name, allow_pickle=True):
  print(f'Loading {file_name}')
  try:
    with open(file_name, 'rb') as f:
      x = np.load(f, allow_pickle=allow_pickle)
  except EOFError:
    print(f"Failed to load data from {f}, file may be corrupted or empty.")
    assert(False)
  return x

def test_saving_and_loading_datasets():
  save_dataset(np.array([1, 2, 3]), np.array([11, 12, 13]), file_name = args.train_set_file)
  x_train, y_train = load_dataset(file_name = args.train_set_file)
  assert((x_train - np.array([1, 2, 3])).sum() == 0)
  assert((y_train - np.array([11, 12, 13])).sum() == 0)
  print(x_train)
  print('----')
  print(y_train)
  print('Pased the test_saving_and_loading_datasets test')
