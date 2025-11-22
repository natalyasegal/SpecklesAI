'''
- test to train format
- slicing
- print stats
'''

import numpy as np
import sys, os
sys.path.append(os.getcwd())
from formatstranslator import test2trainformat

def print_stats(X: np.ndarray, name_str = ''):
  print(f"{name_str} mean = {round(X.mean(), 2)} std = {round(X.std(), 2)} min = {round(X.min(), 2)} max = {round(X.max(), 2)} X.shape={X.shape},")
  
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
