from config import Configuration
from preprocessing import Preprocessing
from models.model_speech_convlstm_tf import train_model

def preprocess_and_train(args_batch_sz, args_n_epochs):
  config = Configuration_Gen(verbose = True)
  if config.be_consistent:
    play_consistent(seed_for_init = config.seed_for_init)
  prep = Preprocessing(config, verbose = config.verbose)
  prep.create_data_set() #videos to frames for train, validation and test sets
  x_train, y_train, x_val, y_val = prep.prepare_train_and_validation_data()
  x_test, y_test = prep.prepare_test_data()
  model_ex3_10, model_history = train_model(config, 9, 8, 
                                            x_train, y_train, 
                                            x_val, y_val,                                        
                                            batch_sz = args_batch_sz, n_epochs = args_n_epochs)

preprocess_and_train(args_batch_sz = 1000, args_n_epochs = 5)
