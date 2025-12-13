'''
- test to train format
- slicing
- print stats
'''

import numpy as np
import sys
from .formatstranslator import test2trainformat

  
def split_by_chunks(arr: np.ndarray, train_n: int = 500, val_n: int = 250,
                    type_axis: int = 0, chunk_axis: int = 1):
    """
    Split an array shaped like (types, chunks, ...) into train/val/test by slicing
    the chunks axis: first train_n, next val_n, rest.

    Parameters
    ----------
    arr : np.ndarray
        Input array; typically shape (T, N, ...), e.g., (2, 5000, 40, 32, 32, 1).
    train_n : int
        Number of chunks per type for the train split.
    val_n : int
        Number of chunks per type for the validation split.
    type_axis : int
        Axis index for 'types' (default 0).
    chunk_axis : int
        Axis index for 'chunks' (default 1).

    Returns
    -------
    train, val, test : np.ndarray
        Slices along the chunks axis with shapes:
        (T, train_n, ...), (T, val_n, ...), (T, N - train_n - val_n, ...)
    """
    arr = np.asarray(arr)
    ndim = arr.ndim
    type_axis %= ndim
    chunk_axis %= ndim

    n_types = arr.shape[type_axis]
    n_chunks = arr.shape[chunk_axis]

    if n_chunks < train_n + val_n:
        raise ValueError(
            f"Not enough chunks ({n_chunks}) for train_n={train_n} and val_n={val_n}."
        )

    # Build slice helpers for the chunks axis
    def sl(start, stop):
        s = [slice(None)] * ndim
        s[chunk_axis] = slice(start, stop)
        return tuple(s)

    train = arr[sl(0, train_n)]
    val   = arr[sl(train_n, train_n + val_n)]
    test  = arr[sl(train_n + val_n, None)]

    return train, val, test


# test_10_m_n.shape == (2, number_of_chunks, 40, 32, 32, 1)
def split_from_start(test_m_n, train_n=500, val_n=250):
  test_m_n_train, test_m_n_val, test_m_n_test = split_by_chunks(
      test_m_n, train_n=train_n, val_n=val_n, type_axis=0, chunk_axis=1
  )
  X_train, y_train = test2trainformat(test_m_n_train, need_to_shuffle_within_category = False)
  X_val, y_val = test2trainformat(test_m_n_val, need_to_shuffle_within_category = False)
  X_test, y_test = test2trainformat(test_m_n_test, need_to_shuffle_within_category = False)

  print((test_m_n_train.shape,test_m_n_val.shape, test_m_n_test.shape))
  print((X_train.shape, X_val.shape, X_test.shape))
  print((y_train.shape, y_val.shape, y_test.shape))
  # Optional sanity checks for your specific case:
  assert test_m_n_train.shape == (2, train_n, 40, 32, 32, 1)
  assert test_m_n_val.shape   == (2, val_n, 40, 32, 32, 1)
  return X_train, X_val, X_test, y_train, y_val, y_test


'''
Loading datasets
'''

def normalize_spatial_exposure_only(x): # for BCI p1, run without this normalization
    """
    Normalize only spatial exposure (per sample and per view),
    preserving temporal and channel dynamics.

    Input:
        x: ndarray of shape (N, 2, T, W, H, 1)
    Output:
        x_norm: same shape, normalized per (sample, view, time, channel)
    """
    if x.ndim != 6:
        raise ValueError("Expected input shape (N, 2, T, W, H, 1)")

    # Transpose to (N, 2, T, H, W, 1) for easier spatial axis access
    x = np.transpose(x, (0, 1, 2, 4, 3, 5))  # shape: (N, 2, T, H, W, 1)

    # Compute mean and std over H and W only (per sample, view, time, channel)
    mean = np.mean(x, axis=(3, 4), keepdims=True)  # shape: (N, 2, T, 1, 1, 1)
    std = np.std(x, axis=(3, 4), keepdims=True) + 1e-6

    # Normalize spatial exposure
    x_norm = (x - mean) / std

    # Return to original shape (N, 2, T, W, H, 1)
    x_norm = np.transpose(x_norm, (0, 1, 2, 4, 3, 5))
    return x_norm

def load_dataset_x(file_name, with_normalization = False):
  print(f'Loading {file_name}')
  try:
    with open(file_name, 'rb') as f:
      x = np.load(f)
  except EOFError:
    print(f"Failed to load data from {f}, file may be corrupted or empty.")
    assert(False)
  return normalize_spatial_exposure_only(x) if with_normalization else x  

def load_dataset_test(file_name):
  x_test_per_category = load_dataset_x(file_name = file_name)
  x_test, y_test = limit_rearrange_and_flatten_s(x_test_per_category, need_to_shuffle_within_category = False)
  return x_test, y_test

def load_test2trainformat(file_name, need_to_shuffle_within_category = False):
  return test2trainformat(load_dataset_x(file_name = file_name), need_to_shuffle_within_category)


def concatenate_train_or_val(x_list, y_list):
  x = np.concatenate(x_list, axis=0)
  y = np.concatenate(y_list, axis=0)
  return x, y
