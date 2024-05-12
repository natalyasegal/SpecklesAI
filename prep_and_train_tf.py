import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration, Configuration_Gen
from dataset import get_or_create_dataset
from evaluation.eval import evaluate_per_chunk, eval_accumulated
from models.model_speech_convlstm_tf import train_model, load_model, set_seed


def main(args):
  config = Configuration_Gen(verbose = True)
  config.split_num = args.split_num
  if config.be_consistent:
    set_seed(seed_for_init = config.seed_for_init, random_seed = args.random_seed)
  x_train, y_train, x_val, y_val, x_test, y_test, x_test_per_category = get_or_create_dataset(config, args, need_to_save = True) 
  model_ex3_10, model_history = train_model(config, 9, 8, 
                                            x_train, y_train, 
                                            x_val, y_val,                                        
                                            batch_sz = args.batch_size, n_epochs = args.epochs)
  model = load_model(config) 
  res_df = evaluate_per_chunk(config, model, x_test, y_test)
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
    parser.add_argument('--shuffle_train_val_within_categories', 
                        action='store_true',
                        help='If specified, suffles samples in train and validation sets within categories, it does not affect the train/val/test split here.')
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


