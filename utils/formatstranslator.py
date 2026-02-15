
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def binarize_lables(labels_orig):
      lb = LabelBinarizer()
      return lb.fit_transform(labels_orig)

def rearrange_input_s(x, need_to_shuffle_within_category, 
                      MAX_CHUNKS_PER_CATEGORY, number_of_classes = 2):
      '''
      Rearranges the input data by randomly shuffling each category and then limits
      the data size to the configured maximum chunks per category.
      '''
      binary_lables = binarize_lables([x for x in range(number_of_classes)])
      y = [[] for _ in range(number_of_classes)]
      print(f'max_chanks={MAX_CHUNKS_PER_CATEGORY}')
      for i in range(number_of_classes): #short loop, as number of categories
          if len(x[i]) == 0:
            print(f'no items in position {i}')
            continue
          print(f' x[{i}] shape is {np.shape(x[i])}')
          if need_to_shuffle_within_category:
            p = np.random.permutation(np.shape(x[i])[0])
            print(' Shaffling within the categories, permutation p shape is {np.shape(p)}')
            x[i] = np.array(x[i])
            x[i] = x[i][p, ::]
          real_max = min(np.shape(x[i])[0], MAX_CHUNKS_PER_CATEGORY)
          x[i] = x[i][0:real_max, ::]
          y[i] = np.full((np.shape(x[i])[0], len(binary_lables[0])), binary_lables[i])
      return x, np.array(y)

def limit_rearrange_and_flatten_s(input_data, need_to_shuffle_within_category, MAX_CHUNKS_PER_CATEGORY, number_of_classes = 2):
    # Rearrange the input data and get the corresponding labels
    rearranged_data, labels = rearrange_input_s(input_data, need_to_shuffle_within_category, MAX_CHUNKS_PER_CATEGORY, number_of_classes=number_of_classes)

    # Concatenate the rearranged data and labels along the first axis
    output_data = np.concatenate(rearranged_data, axis=0)
    output_labels = np.concatenate(labels, axis=0)

    # Return the concatenated and rearranged data along with their labels
    return output_data, output_labels

def test2trainformat(input_data, need_to_shuffle_within_category = False, MAX_CHUNKS_PER_CATEGORY = 500000):
    if input_data is None:
        return None, None

    number_of_classes = len(input_data)
    x, y = limit_rearrange_and_flatten_s(input_data, need_to_shuffle_within_category, MAX_CHUNKS_PER_CATEGORY, number_of_classes=number_of_classes)
    return x, y #subject_normalize_preserve_time(x), y

def make_x_per_category(X, y, class_order=(0, 1)):
    X = np.asarray(X)
    y = np.asarray(y)
    # if one-hot, reduce to argmax
    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    else:
        y = y.ravel().astype(int)
    return [X[y == c] for c in class_order]
