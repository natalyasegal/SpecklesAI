import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration, Configuration_Gen
from preprocessing.preprocessing import Preprocessing
from visualization.visualization import visualize_speckles
from utils.utils import save_dataset_x, save_dataset, load_dataset_x, load_dataset


def get_or_create_dataset(config, args, need_to_save):
  '''
  Creates or loads train, validation and test datasets
  Datasets are created from videos files by creating arrays od 3d tensors (X x Y x Temporal chunk)
  '''
  prep = Preprocessing(config, verbose = config.verbose)
  if not args.read_stored_dataset:
    prep.create_data_set() #videos to frames for train, validation and test sets
    x_train, y_train, x_val, y_val = prep.prepare_train_and_validation_data(need_to_shuffle_within_category = args.shuffle_train_val_within_categories)
    x_test, y_test, x_test_per_category = prep.prepare_test_data()
    if need_to_save:
      save_dataset(x_train, y_train, file_name = args.train_set_file)
      save_dataset(x_val, y_val, file_name = args.validation_set_file)
      save_dataset_x(x_test_per_category, file_name = args.test_set_per_category_file)
  else:
    x_train, y_train = load_dataset(file_name = args.train_set_file)
    x_val, y_val = load_dataset(file_name = args.validation_set_file)
    x_test_per_category = load_dataset_x(file_name = args.test_set_per_category_file)
    x_test, y_test = prep.limit_rearrange_and_flatten(x_test_per_category, need_to_shuffle_within_category = False)
  if config.create_images:
    visualize_speckles(x_train, save_path = 'speckles_sample.png', please_also_show = False)
  return x_train, y_train, x_val, y_val, x_test, y_test, x_test_per_category
  
  def get_or_create_dataset_train_and_val(config, args, need_to_save):
    '''
    Creates or loads train, validation datasets
    Datasets are created from videos files by creating arrays od 3d tensors (X x Y x Temporal chunk)
    '''
    prep = Preprocessing(config, verbose = config.verbose)
    if not args.read_stored_dataset:
      prep.create_data_set() #videos to frames for train, validation and test sets
      x_train, y_train, x_val, y_val = prep.prepare_train_and_validation_data(need_to_shuffle_within_category = args.shuffle_train_val_within_categories)
      if need_to_save:
        save_dataset(x_train, y_train, file_name = args.train_set_file)
        save_dataset(x_val, y_val, file_name = args.validation_set_file)   
    else:
      x_train, y_train = load_dataset(file_name = args.train_set_file)
      x_val, y_val = load_dataset(file_name = args.validation_set_file)
    if config.create_images:
      visualize_speckles(x_train, save_path = 'speckles_sample.png', please_also_show = False)
    return x_train, y_train, x_val, y_val

def get_or_create_dataset_test(config, args, need_to_save):
  '''
  Creates or loads test datasets
  Datasets are created from videos files by creating arrays od 3d tensors (X x Y x Temporal chunk)
  '''
  prep = Preprocessing(config, verbose = config.verbose)
  if not args.read_stored_dataset:
    prep.create_data_set() #videos to frames for train, validation and test sets
    x_test, y_test, x_test_per_category = prep.prepare_test_data()
    if need_to_save:
      save_dataset_x(x_test_per_category, file_name = args.test_set_per_category_file)
  else:
    x_test_per_category = load_dataset_x(file_name = args.test_set_per_category_file)
    x_test, y_test = prep.limit_rearrange_and_flatten(x_test_per_category, need_to_shuffle_within_category = False)
  return x_test, y_test, x_test_per_category
