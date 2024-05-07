import numpy as np
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from config.config import Configuration, Configuration_Gen
from preprocessing.preprocessing import Preprocessing
from evaluation.eval import evaluate_per_chunk
from models.model_speech_convlstm_tf import train_model, play_consistent
from visualization.visualization import visualize_speckles

def save_dataset_x(x, file_name):
  with open(file_name, 'wb') as f:
      np.save(f, x)

def save_dataset(x, y, file_name):
  with open(file_name, 'wb') as f:
      np.save(f, x)
      np.save(f, y)

def load_dataset(file_name):
  with open(file_name, 'rb') as f:
    x = np.load(f)
    y = np.load(f)
  return x, y

def get_or_create_dataset(config, need_to_create = True, need_to_save = False):
  if need_to_create:
    prep = Preprocessing(config, verbose = config.verbose)
    prep.create_data_set() #videos to frames for train, validation and test sets
    x_train, y_train, x_val, y_val = prep.prepare_train_and_validation_data()
    x_test, y_test, x_test_per_category = prep.prepare_test_data()
    if need_to_save:
      save_dataset(x_train, y_train, file_name = 'train_set.npy')
      save_dataset(x_val, y_val, file_name = 'validation_set.npy')
      save_dataset(x_test, y_test, file_name = 'test_set.npy')
      save_dataset_x(x_test_per_category, file_name = 'test_per_category.npy')
  else:
    x_train, y_train = load_dataset(file_name = 'train_set.npy')
    x_val, y_val = load_dataset(file_name = 'validation_set.npy')
    x_test, y_test = load_dataset(file_name = 'test_set.npy')
    x_test_per_category = load_dataset(file_name = 'test_per_category.npy')
  if config.create_images:
    visualize_speckles(x_train, save_path = 'speckles_sample.png', please_also_show = False)
  return x_train, y_train, x_val, y_val, x_test, y_test, x_test_per_category

def preprocess_and_train(args_batch_sz, args_n_epochs):
  config = Configuration_Gen(verbose = True)
  if config.be_consistent:
    play_consistent(seed_for_init = config.seed_for_init)
  x_train, y_train, x_val, y_val, x_test, y_test, x_test_per_category = get_or_create_dataset(
      config, need_to_create = True, need_to_save = False) 
  model_ex3_10, model_history = train_model(config, 9, 8, 
                                            x_train, y_train, 
                                            x_val, y_val,                                        
                                            batch_sz = args_batch_sz, n_epochs = args_n_epochs)
  model = tf.keras.models.load_model(config.models_path)
  res_df = evaluate_per_chunk(config, model, x_test, y_test)
  return model, model_history, x_test_per_category, x_test, y_test, x_train, y_train, x_val, y_val, config, res_df

#TODO: create main:

model, model_history, x_test_per_category, x_test, y_test, x_train, y_train, x_val, y_val, config, res_df = preprocess_and_train(
    args_batch_sz=1000, args_n_epochs=25)
ref_df_a = eval_accumulated(config, model, x_test_per_category, num_of_chunks_to_aggregate = 25)



