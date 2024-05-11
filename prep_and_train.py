import argparse
import numpy as np
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration, Configuration_Gen
from preprocessing.preprocessing import Preprocessing
from evaluation.eval import evaluate_per_chunk, eval_accumulated
from models.model_speech_convlstm_tf import train_model, load_model, play_consistent
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
    x_train, y_train, x_val, y_val = prep.prepare_train_and_validation_data(need_to_shuffle_within_category = args.shuffle_test_val_within_categories)
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

def preprocess_and_train(args):
  config = Configuration_Gen(verbose = True)
  config.split_num = args.split_num
  if config.be_consistent:
    play_consistent(seed_for_init = config.seed_for_init, random_seed = args.random_seed)
  x_train, y_train, x_val, y_val, x_test, y_test, x_test_per_category = get_or_create_dataset(config, args, need_to_save = True) 
  model_ex3_10, model_history = train_model(config, 9, 8, 
                                            x_train, y_train, 
                                            x_val, y_val,                                        
                                            batch_sz = args.batch_size, n_epochs = args.epochs)
  model = load_model(config) 
  res_df = evaluate_per_chunk(config, model, x_test, y_test)
  return model, model_history, x_test_per_category, x_test, y_test, x_train, y_train, x_val, y_val, config, res_df


def main(args):
  model, model_history, x_test_per_category, x_test, y_test, x_train, y_train, x_val, y_val, config, res_df = preprocess_and_train(args)
  ref_df_a = eval_accumulated(config, model, x_test_per_category, num_of_chunks_to_aggregate = args.num_of_chunks_to_aggregate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=1000)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=25)
    parser.add_argument('--num_of_chunks_to_aggregate',
                        help='num_of_chunks_to_aggregate',
                        type=int,
                        default=25)
    parser.add_argument('--random_seed',
                        help='seed for python random, used in shafling, does not affect division into train, validation and test',
                        type=int,
                        default=2)
    parser.add_argument('--split_num',
                        help='If provided, overwrites the parameter in config.py with the same name, given the split number, the actual split will be read from a configuation file, you provide',
                        type=int,
                        default=6)

    parser.add_argument('--read_stored_dataset', 
                        action='store_true',
                        help='If specified, read parsed frame chunks for dataset; otherwise, create them.')
    parser.add_argument('--shuffle_test_val_within_categories', 
                        action='store_true',
                        help='If specified, suffles samples in train and validation sets within categories, it does not affect the train/val/trst split.')
    parser.add_argument('--train_set_file',
                        help='train parsed data arranges in chanks, chank size is designated in config.py',
                        type=str,
                        default='train_set.npy')
    parser.add_argument('--validation_set_file',
                        help='validation parsed data arranges in chanks, chank size is designated in config.py',
                        type=str,
                        default='validation_set.npy')
    parser.add_argument('--test_set_per_category_file',
                        help='test parsed data arranges in chanks, given by category, chank size is designated in config.py',
                        type=str,
                        default='test_per_category.npy')  
   
    args = parser.parse_args()
    main(args)



