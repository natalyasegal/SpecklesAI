import argparse
import sys
import os
import numpy as np
import random
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration_PerSubjExperiment
from preprocessing.preprocessing import Preprocessing, unison_shuffled_copies
from utils.utils import save_dataset_x, save_dataset, load_dataset_x, load_dataset

''' 
Videos -> np arrayss of chunks of frames
This part does not require GPU, run it if you prefer to decouple preprocessing from the GPU intensive training

Prepares datasets for per subject experiments
'''
def prepare_train_and_validation_data_per_subj_experiment(prep, need_to_shuffle_within_category):
    '''train and validation set preprocessing'''
    #assert(prep.__have_train_set_parsed)
    prep.log(f'prep.config.train_subjects = {prep.config.train_subjects}, prep.config.train_dates = {prep.config.train_dates}, prep.config.val_limit = {prep.config.val_limit}')
    x_train_per_category = prep.prep_frames_lists(prep.config.train_dates, prep.config.train_subjects, 'train')
   
    """ Use part of the morning sesuence of chunks for validation """
    x_val_per_category = x_train_per_category[ :, 0:prep.config.val_limit ]
    x_train_per_category = x_train_per_category[ :, prep.config.val_limit: ]
  
    x_train, y_train = prep.limit_rearrange_and_flatten(x_train_per_category, need_to_shuffle_within_category)
    prep.log(f'x_train shape is {np.shape(x_train)}, y_train shape is {np.shape(y_train)}')

    x_val, y_val = prep.limit_rearrange_and_flatten(x_val_per_category, need_to_shuffle_within_category)
    prep.log(f'x_val shape is {np.shape(x_val)}, y_val shape is {np.shape(y_val)}')
  
    '''shaffle train and validation sets, data and lables are shaffled together'''
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_val, y_val = unison_shuffled_copies(x_val, y_val)
    return x_train, y_train, x_val, y_val
  
def create_per_subj_datasets(args):
  number_of_subj = 6
  for i in range(number_of_subj):
    config = Configuration_PerSubjExperiment(i+1, verbose = True)  
    if config.be_consistent:
      np.random.seed(config.seed_for_init)  # Set seed for NumPy operations to ensure reproducibility
      random.seed(args.random_seed)
    prep = Preprocessing(config, verbose = config.verbose)
    prep.create_data_set() #videos to frames for train, validation and test sets
    x_train, y_train, x_val, y_val = prepare_train_and_validation_data_per_subj_experiment(prep, need_to_shuffle_within_category = args.shuffle_train_val_within_categories)
    x_test, y_test, x_test_per_category = prep.prepare_test_data()
    
    save_dataset(x_train, y_train, file_name = f'{args.train_set_file}_{str(config.split_num)}.npy')
    save_dataset(x_val, y_val, file_name = f'{args.validation_set_file}_{str(config.split_num)}.npy')
    save_dataset_x(x_test_per_category, file_name = f'{args.test_set_per_category_file}_{str(config.split_num)}.npy') 


def create_sanity_test_set(args):
    random_seed = 9 
    subj_num = 6
    config = Configuration_PerSubjExperiment(subj_num, verbose = True)  
    config.set_split([], [], [] , [], args.test_set, config.test_subjects)
    print(f' test {config.test_dates} {config.test_subjects}')

    if config.be_consistent:
      np.random.seed(config.seed_for_init)  # Set seed for NumPy operations to ensure reproducibility
      random.seed(random_seed)
    prep = Preprocessing(config, verbose = config.verbose)
    prep.create_data_set() #videos to frames for train, validation and test sets
    x_test, y_test, x_test_per_category = prep.prepare_test_data()
    save_dataset_x(x_test_per_category, file_name = f'{args.test_set_per_category_file}_{str(config.split_num)}.npy') 

def main(args):
    if args.create_only_sanity_test_set:
        create_sanity_test_set(args)
    else:
        create_per_subj_datasets(args)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')

    parser.add_argument('--random_seed',
                        help='seed for python random, used in shafling, does not affect division into train, validation and test',
                        type=int,
                        default=2)
    parser.add_argument('--shuffle_train_val_within_categories', 
                        action='store_true',
                        help='If specified, suffles samples in train and validation sets within categories, it does not affect the train/val/test split here.')

    parser.add_argument('--create_only_sanity_test_set', 
                        action='store_true',
                        help='If specified, create only a tests set for forehead sanity test for subj 6.')

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

    parser.add_argument('--test_set',
                        help='a list of directories containing video files for the test set',
                        type=list,
                        default=['08012024_day1_forehead', '08012024_day1_1_forehead']) 
    test_set
   
    args = parser.parse_args()
    main(args)
