from config import Configuration
from preprocessing import Preprocessing
from evaluation.eval import evaluate_per_chunk
from models.model_speech_convlstm_tf import train_model
import numpy as np

def save_dataset(x, y, file_name = 'train_set.npy'):
  with open(file_name, 'wb') as f:
      np.save(f, x)
      np.save(f, y)

def load_dataset(file_name = 'train_set.npy'):
  with open(file_name, 'rb') as f:
    x = np.load(f)
    y = np.load(f)
  return x, y

def preprocess_and_train(args_batch_sz, args_n_epochs, create_images = True):
  config = Configuration_Gen(verbose = True)
  if config.be_consistent:
    play_consistent(seed_for_init = config.seed_for_init)
  prep = Preprocessing(config, verbose = config.verbose)
  prep.create_data_set() #videos to frames for train, validation and test sets
  x_train, y_train, x_val, y_val = prep.prepare_train_and_validation_data()
  save_dataset(x_train, y_train, file_name = 'train_set.npy')
  save_dataset(x_val, y_val, file_name = 'validation_set.npy')
  x_test, y_test = prep.prepare_test_data()
  save_dataset(x_test, y_test, file_name = 'test_set.npy')
  if create_images:
    visualize_speckles(x_train, save_path = 'speckles_sample.png', please_also_show = False)
  model_ex3_10, model_history = train_model(config, 9, 8, 
                                            x_train, y_train, 
                                            x_val, y_val,                                        
                                            batch_sz = args_batch_sz, n_epochs = args_n_epochs)
  model = tf.keras.models.load_model(config.models_path)
  res_df = evaluate_per_chunk(config, model, x_test, y_test, create_images = True)
  return model, model_history, x_test, y_test, x_train, y_train, x_val, y_val, config, res_df

model, model_history, x_test, y_test, x_train, y_train, x_val, y_val, config, res_df = preprocess_and_train(
    args_batch_sz=1000, args_n_epochs=50)

