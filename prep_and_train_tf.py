import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration, Configuration_Gen, Configuration_PerSubjExperiment
from dataset import get_or_create_dataset, get_or_create_dataset_train_and_val, get_or_create_dataset_test
from evaluation.eval import evaluate_per_chunk, eval_accumulated
from models.model_speech_convlstm_tf import train_model, load_model, set_seed


def main(args):
  if args.use_per_subj_config:
    config = Configuration_PerSubjExperiment(args.split_num, verbose = True) 
  else:
    config = Configuration_Gen(args.split_num, args.config_file, verbose = True)
  if config.be_consistent:
    set_seed(seed_for_init = config.seed_for_init, random_seed = args.random_seed)

  print(f'args.overwrite_nclasses = {args.overwrite_nclasses}')
  if args.overwrite_nclasses:
    config.number_of_classes = args.nclasses
    config.lables_categories = [x for x in range(config.number_of_classes)]
    config.binary_lables = binarize_lables(config.lables_categories)
    print(f' number_of_classes = {config.number_of_classes}\n binary_lables={config.binary_lables}')
  
  if not args.save_RAM:
    x_train, y_train, x_val, y_val, x_test, y_test, x_test_per_category = get_or_create_dataset(config, args, need_to_save = True) 
    model_ex3_10, model_history = train_model(config, args.sz_conv, args.sz_dense, 
                                              x_train, y_train, 
                                              x_val, y_val,                                        
                                              batch_sz = args.batch_size, n_epochs = args.epochs, metric_to_monitor = args.metric_to_monitor)
    model = load_model(config) 
    res_df = evaluate_per_chunk(config, model, x_test, y_test)
    ref_df_a = eval_accumulated(config, model, x_test_per_category, num_of_chunks_to_aggregate = args.num_of_chunks_to_aggregate)
  else:
    x_train, y_train, x_val, y_val = get_or_create_dataset_train_and_val(config, args, need_to_save = True) 
    model_ex3_10, model_history = train_model(config, args.sz_conv, args.sz_dense, 
                                                x_train, y_train, 
                                                x_val, y_val,                                        
                                                batch_sz = args.batch_size, n_epochs = args.epochs, metric_to_monitor = args.metric_to_monitor)
    del x_train, y_train, x_val, y_val
    x_test, y_test, x_test_per_category = get_or_create_dataset_test(config, args, need_to_save = True)
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
                        default=9)
    parser.add_argument('--split_num',
                        help='If provided, overwrites the parameter in config.py with the same name, given the split number, the actual split will be read from a configuation file, you provide',
                        type=int,
                        default=6)

    parser.add_argument('--sz_conv',
                        help='If provided, overwrites the size of the convLSTM output',
                        type=int,
                        default=9)
    parser.add_argument('--sz_dense',
                        help='If provided, overwrites the size of the dense layer output',
                        type=int,
                        default=8)
    parser.add_argument('--config_file',
                        help='configuration file name',
                        type=str,
                        default='SpecklesAI/config/config_files/sample_gen_split.yaml')  
    parser.add_argument('--use_per_subj_config', 
                        action='store_true',
                        help='If specified, uses per_subj experiment config option, this affects mostly printouts. Use pre-created datasets with this option.')
    parser.add_argument('--save_RAM', 
                        action='store_true',
                        help='Save RAM memory.')
    parser.add_argument('--read_stored_dataset', 
                        action='store_true',
                        help='If specified, read parsed frame chunks for dataset; otherwise, create them.')
    parser.add_argument('--overwrite_nclasses', 
                        action='store_false',
                        help='If specified, overwrite the number of classes in config.')
    parser.add_argument('--nclasses',
                        help='If provided, overwrites the number of classes',
                        type=int,
                        default=2)
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
    parser.add_argument('--metric_to_monitor',
                        help='metric to monitor during training',
                        type=str,
                        default='val_accuracy') 
   
    args = parser.parse_args()
    main(args)



