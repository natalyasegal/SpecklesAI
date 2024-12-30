'''
Sample usage:
python -u SpecklesAI/prepare_test_sets.py --split_num 1 --random_seed 9  --test_set_per_category_file test_per_category_split_midday_
'''

import argparse
import sys
import os
import numpy as np
import random
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration, Configuration_Test
from preprocessing.preprocessing import Preprocessing
from utils.utils import save_dataset_x, save_dataset, load_dataset_x, load_dataset
     
def handle_one_split(split_num):
    config = Configuration_Test(split_num, verbose = True)
    print(f'handle_one_split: split_num = {handle_one_split}.')
    if config.be_consistent:
      np.random.seed(config.seed_for_init)  # Set seed for NumPy operations to ensure reproducibility
      random.seed(args.random_seed)
    prep = Preprocessing(config, verbose = config.verbose)
    prep.create_test_data_set() #videos to frames for test sets
    x_test, y_test, x_test_per_category = prep.prepare_test_data()
    save_dataset_x(x_test_per_category, file_name = f'{args.test_set_per_category_file}_{str(config.split_num)}.npy') 

''' 
Videos -> np arrayss of chunks of frames
This part does not require GPU, run it if you prefer to decouple preprocessing from the GPU intensive training
'''
def main(args):
  config = Configuration_Test(1, verbose = True)  #just to parse the splits file
  number_of_splits = len(config.sample_splits)
  if args.split_num > -1: #set to something other then default
    if args.split_num <= number_of_splits:
      handle_one_split(args.split_num)
  else:
    for i in range(number_of_splits):
      handle_one_split(i+1)
      
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
    parser.add_argument('--test_set_per_category_file',
                        help='test parsed data arranges in chanks, given by category, chank size is designated in config.py',
                        type=str,
                        default='test_per_category_')  
   
    args = parser.parse_args()
    main(args)
