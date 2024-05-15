import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration, Configuration_Gen, Configuration_PerSubjExperiment     
from preprocessing.preprocessing import Preprocessing
from evaluation.eval import evaluate_per_chunk, eval_accumulated
from models.model_speech_convlstm_tf import load_model, set_seed
from visualization.visualization import visualize_speckles
from utils.utils import save_dataset_x, load_dataset_x

''' Place the trained model in predefined directory and run on test set to inference and evaluate'''

def get_or_create_test_dataset(config, args, need_to_save):
  '''
  Creates or loads test dataset
  Dataset is created from videos files by creating arrays od 3d tensors (X x Y x Temporal chunk)
  '''
  prep = Preprocessing(config, verbose = config.verbose)
  if not args.read_stored_dataset:
    prep.create_test_data_set() #videos to frames for train, validation and test sets
    x_test, y_test, x_test_per_category = prep.prepare_test_data()
    if need_to_save:
      save_dataset_x(x_test_per_category, file_name = args.test_set_per_category_file)
  else:
    x_test_per_category = load_dataset_x(file_name = args.test_set_per_category_file)
    x_test, y_test = prep.limit_rearrange_and_flatten(x_test_per_category, need_to_shuffle_within_category = False)
  if config.create_images:
    visualize_speckles(x_test, save_path = 'speckles_sample.png', please_also_show = False)
  return x_test, y_test, x_test_per_category


def main(args):
  if args.use_per_subj_config:
    config = Configuration_PerSubjExperiment(args.split_num, verbose = True) 
  else:
    config = Configuration_Gen(args.split_num, verbose = True)
    
  if config.be_consistent:
    set_seed(seed_for_init = config.seed_for_init, random_seed = args.random_seed)
  x_test, y_test, x_test_per_category = get_or_create_test_dataset(config, args, need_to_save = True) 
  model = load_model(config) 
  res_df = evaluate_per_chunk(config, model, x_test, y_test)
  ref_df_a = eval_accumulated(config, model, x_test_per_category, num_of_chunks_to_aggregate = args.num_of_chunks_to_aggregate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')

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
    parser.add_argument('--test_set_per_category_file',
                        help='test parsed data arranges in chanks, given by category, chank size is designated in config.py',
                        type=str,
                        default='test_per_category.npy')

    parser.add_argument('--use_per_subj_config', 
                        action='store_true',
                        help='If specified, uses per_subj experiment config option, this affects mostly printouts. Use pre-created datasets with this option.')

   
    args = parser.parse_args()
    main(args)
