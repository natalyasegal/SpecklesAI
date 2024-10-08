import argparse
import sys
import os
import numpy as np
import random
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration, Configuration_Gen
from preprocessing.preprocessing import Preprocessing
from utils.utils import save_dataset_x, save_dataset, load_dataset_x, load_dataset

''' 
Videos -> np arrayss of chunks of frames
This part does not require GPU, run it if you prefer to decouple preprocessing from the GPU intensive training
'''

def main(args):
  config = Configuration_Gen(1, verbose = True)  #just to parse the splits file
  number_of_splits = len(config.sample_splits)
  for i in range(number_of_splits):
    config = Configuration_Gen(i+1, verbose = True)  
    if config.be_consistent:
      np.random.seed( config.seed_for_init)  # Set seed for NumPy operations to ensure reproducibility
      random.seed(args.random_seed)
    prep = Preprocessing(config, verbose = config.verbose)
    prep.create_data_set() #videos to frames for train, validation and test sets
    x_train, y_train, x_val, y_val = prep.prepare_train_and_validation_data(need_to_shuffle_within_category = args.shuffle_train_val_within_categories)
    x_test, y_test, x_test_per_category = prep.prepare_test_data()
    
    save_dataset(x_train, y_train, file_name = f'{args.train_set_file}_{str(config.split_num)}.npy')
    save_dataset(x_val, y_val, file_name = f'{args.validation_set_file}_{str(config.split_num)}.npy')
    save_dataset_x(x_test_per_category, file_name = f'{args.test_set_per_category_file}_{str(config.split_num)}.npy') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')

    parser.add_argument('--random_seed',
                        help='seed for python random, used in shafling, does not affect division into train, validation and test',
                        type=int,
                        default=2)
    parser.add_argument('--shuffle_train_val_within_categories', 
                        action='store_true',
                        help='If specified, suffles samples in train and validation sets within categories, it does not affect the train/val/test split here.')
    parser.add_argument('--train_set_file',
                        help='train parsed data arranges in chanks, chank size is designated in config.py',
                        type=str,
                        default='train_set_')
    parser.add_argument('--validation_set_file',
                        help='validation parsed data arranges in chanks, chank size is designated in config.py',
                        type=str,
                        default='validation_set_')
    parser.add_argument('--test_set_per_category_file',
                        help='test parsed data arranges in chanks, given by category, chank size is designated in config.py',
                        type=str,
                        default='test_per_category_')  
   
    args = parser.parse_args()
    main(args)
