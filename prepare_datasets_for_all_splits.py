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
     
def handle_one_split(split_num, config_file_name):
    config = Configuration_Gen(split_num, config_file_name, verbose = True)
    print(f'handle_one_split: split_num = {handle_one_split} and config_file_name is {config_file_name}.')
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

''' 
Videos -> np arrayss of chunks of frames
This part does not require GPU, run it if you prefer to decouple preprocessing from the GPU intensive training
'''
def main(args):
  config = Configuration_Gen(1, args.config_file, verbose = True)  #just to parse the splits file
  number_of_splits = len(config.sample_splits)
  if args.split_num > -1: #set to something other then default
    if args.split_num <= number_of_splits:
      handle_one_split(args.split_num, args.config_file)
  else:
    for i in range(number_of_splits):
      handle_one_split(i+1, args.config_file)
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')

    parser.add_argument('--random_seed',
                        help='seed for python random, used in shafling, does not affect division into train, validation and test',
                        type=int,
                        default=2)
    parser.add_argument('--split_num',
                        help='one split to create the sets for, used for initial testing, defaults to non-existent value -1',
                        type=int,
                        default=-1)
    parser.add_argument('--shuffle_train_val_within_categories', 
                        action='store_true',
                        help='If specified, suffles samples in train and validation sets within categories, it does not affect the train/val/test split here.')
    parser.add_argument('--config_file',
                        help='configuration file name',
                        type=str,
                        default='SpecklesAI/config/config_files/sample_gen_split.yaml')    
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
