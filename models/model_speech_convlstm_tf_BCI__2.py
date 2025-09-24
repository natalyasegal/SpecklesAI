import numpy as np
import random
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import Sequential, BatchNormalization, GroupNormalization, ConvLSTM2D, Dropout, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from keras.layers import TFSMLayer, Input
from keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, CSVLogger, TensorBoard)


#tf.keras.mixed_precision.set_global_policy("mixed_float16")
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("Could not set memory growth:", e)
        
tf.config.optimizer.set_jit(False)  # avoid XLA surprises on Colab


'''The model works with video chunks of recorded speckle pappern'''

def set_seed(seed_for_init = 1, random_seed = 9):
    np.random.seed(seed_for_init)  # Set seed for NumPy operations to ensure reproducibility
    random.seed(random_seed)
    
    ''' specific for Tensorflow: '''

    # Ensure deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    tf.random.set_seed(seed_for_init)  # Set seed for TensorFlow operations to ensure reproducibility

# ---- Data loader
class NumpySequence(Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        self.x, self.y = x, y
        self.bs, self.shuffle = batch_size, shuffle
        self.idx = np.arange(len(x))
        self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.x)/self.bs))
    def __getitem__(self, i):
        sl = slice(i*self.bs, (i+1)*self.bs)
        ids = self.idx[sl]
        return self.x[ids], self.y[ids]
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.idx)
# ----


def define_model(config, sz_conv, sz_dense, metric_to_monitor='accuracy', week_lables = True):
    """Defines a ConvLSTM2D model for classification tasks."""
    print("---trying GroupNormalization(groups=4)")
    model = Sequential([
        #BatchNormalization(),
        GroupNormalization(groups=1),
        tf.keras.layers.LayerNormalization(axis=[2,3,4], name="pre_convlstm_ln"),
        ConvLSTM2D(
            filters=sz_conv,
            kernel_size=(3, 3),
            kernel_constraint=max_norm(2.),
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=0),
            return_sequences=False,  # Final ConvLSTM before flatten #True, #False,
            data_format="channels_last",
            input_shape=(None, config.frame_size_y, config.frame_size_x, 1)
        ),
        BatchNormalization(), 
        Dropout(0.5),
        Flatten(),
        Dense(
            sz_dense,
            kernel_constraint=max_norm(2.),
            kernel_regularizer='l2',
            activation="relu"
        ),
        tf.keras.layers.LayerNormalization(),
        Dropout(0.5)
    ])

    # Add output layer and compile the model based on number of classes
    output_units = config.number_of_classes if config.number_of_classes > 2 else config.number_of_classes - 1
    activation = "softmax" if config.number_of_classes > 2 else "sigmoid"

    model.add(Dense(output_units, kernel_constraint=max_norm(2.), activation=activation))
    
    if str(metric_to_monitor).lower() in ('auc', 'val_auc'):
        metrics = [tf.keras.metrics.AUC(name='auc'), 'accuracy']  
    elif str(metric_to_monitor).lower() in ('accuracy', 'val_accuracy', 'acc', 'val_acc'):
        metrics = ['accuracy']
    else:
        metrics = ['accuracy']  # safe default
    
    loss = 'categorical_crossentropy' if config.number_of_classes > 2 else 'binary_crossentropy'
    
    model.compile(loss=loss, optimizer="adam", metrics=metrics)

    return model, metrics
    
def save_auc_plot(config, history, save_path = 'auc.png', show = False):
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    data = { 'auc': history.history['auc'],
              'val_auc': history.history['val_auc']
           }
    acc_df = pd.DataFrame(data)
    acc_df.to_csv(f'{config.model_name}_auc.csv', index=False)
    print(f'Results saved to {config.model_name}_auc.csv')
    plt.clf()


def save_accuracy_plot(config, history, save_path = 'accuracy.png', show = False):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    data = { 'accuracy': history.history['accuracy'],
              'val_accuracy': history.history['val_accuracy']
           }
    acc_df = pd.DataFrame(data)
    acc_df.to_csv(f'{config.model_name}_accuracy.csv', index=False)
    print(f'Results saved to {config.model_name}_accuracy.csv')
    plt.clf()

def save_loss_plot(config, history, save_path = 'loss.png', show = False):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    data = { 'loss': history.history['loss'],
              'val_loss': history.history['val_loss']
           }
    loss_df = pd.DataFrame(data)
    loss_df.to_csv(f'{config.model_name}_loss.csv', index=False)
    print(f'Results saved to {config.model_name}_loss.csv')
    plt.clf()

'''
Helper functions to support week lables:
'''

def make_sample_weights(y, class_weights):
    """
    y: (N,) ints OR (N,K) one-hot
    class_weights: dict {class_index: weight}
    returns: (N,) float32 sample weights
    """
    y = np.asarray(y)
    if y.ndim == 1:  # integer labels
        # If pause label is '3', remap to index 2 for 3-class training
        # (comment this line out if your pause already equals 2)
        y = np.where(y == 3, 2, y).astype(int)
        w = np.vectorize(lambda c: class_weights.get(int(c), 1.0))(y)
        return w.astype(np.float32)
    else:            # one-hot (N,K)
        wvec = np.array([class_weights.get(i, 1.0) for i in range(y.shape[1])], dtype=np.float32)
        return (y @ wvec).astype(np.float32)


def build_weak_strong_sample_weights(config, y_train, y_val):
    """
    Create per-sample weights where classes 0/1 are 'weak' and pause (class 2) is 'strong'.
    If your dataset encodes pause as '3', your existing make_sample_weights() already
    remaps it to index 2 (as shown earlier).

    Args:
        config: has optional attributes weak_weight (default 0.5) and pause_weight (default 2.0)
        y_train: (N,) ints or (N,3) one-hot
        y_val:   (M,) ints or (M,3) one-hot (optional)

    Returns:
        sw_train: (N,) float32 sample weights
        sw_val:   (M,) float32 sample weights or None if y_val is None
        class_weights: dict {0: weak_w, 1: weak_w, 2: pause_w}
    """
    weak_w  = getattr(config, "weak_weight", 0.5)   # tweakable
    pause_w = getattr(config, "pause_weight", 5)  # tweakable #2
    print(f'weak_w {weak_w}, pause_w {pause_w}')
    class_weights = {0: weak_w, 1: weak_w, 2: pause_w}

    sw_train = make_sample_weights(y_train, class_weights)
    sw_val   = make_sample_weights(y_val,   class_weights) if y_val is not None else None
    return sw_train, sw_val, class_weights

def pause_biased_categorical_crossentropy(alpha=0.2, pause_index=2, from_logits=False):
    """
    For non-pause labels, shift 'alpha' probability mass to the pause class.
    pause (y[..., pause_index]==1) stays one-hot.
    """
    def loss(y_true, y_pred):
        # ensure one-hot targets
        K = tf.shape(y_pred)[-1]
        if y_true.shape.rank is not None and y_true.shape.rank > 1 and y_true.shape[-1] == y_pred.shape[-1]:
            y_oh = tf.cast(y_true, y_pred.dtype)
        else:
            y_oh = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=K, dtype=y_pred.dtype)

        # mask: 1 for non-pause rows, 0 for pause rows
        non_pause = 1.0 - y_oh[..., pause_index:pause_index+1]   # shape (N,1)

        # one-hot vector for pause class, broadcasted
        e_pause = tf.one_hot(pause_index, K, dtype=y_pred.dtype)  # (K,)
        e_pause = tf.reshape(e_pause, [1, K])                     # (1,K)

        # y_true_mod = y_true*(1 - alpha) + alpha*e_pause  (only for non-pause rows)
        y_true_mod = y_oh * (1.0 - alpha * non_pause) + (alpha * non_pause) * e_pause

        # standard categorical CE on modified targets
        ce = tf.keras.losses.categorical_crossentropy(
            y_true_mod, y_pred, from_logits=from_logits, label_smoothing=0.0
        )
        return tf.reduce_mean(ce)
    return loss


''' train '''

def train_model(config, sz_conv, sz_dense, x_train, y_train, x_val, y_val,
                    batch_sz, n_epochs, metric_to_monitor='auc', week_lables = True):
        """Trains the model and saves the best model based on validation accuracy."""
    
        # Ensure the model save path exists
        os.makedirs(config.models_path, exist_ok=True)
    
        # Define the model save path with the .keras extension
        config.models_path = os.path.join(config.models_path, 'best_model.keras')
    
        model_checkpoint_callback = ModelCheckpoint(
            filepath=config.models_path,
            save_weights_only=False,
            monitor=metric_to_monitor,
            mode='max',
            save_best_only=True
        )
    
        model, metrics = define_model(config, sz_conv, sz_dense, metric_to_monitor=metric_to_monitor, week_lables=week_lables)
    
        # --- stable Sequence option instead of raw arrays
        #train_gen = NumpySequence(x_train, y_train, batch_sz, shuffle=True)
        #val_gen   = NumpySequence(x_val,   y_val,   batch_sz, shuffle=False)
        
        if week_lables and config.number_of_classes > 2:
            sw_train, sw_val, class_weights = build_weak_strong_sample_weights(config, y_train, y_val)
            model_history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val, sw_val) if sw_val is not None else (x_val, y_val),
                batch_size=batch_sz,
                epochs=n_epochs,callbacks=[model_checkpoint_callback],
                sample_weight=sw_train
            )
        else:
            model_history = model.fit(
                #train_gen,validation_data=val_gen,
                x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_sz, #
                epochs=n_epochs,callbacks=[model_checkpoint_callback] #,
                ##workers=0,    # do not use those parameters in Colab
                ##use_multiprocessing=False
            )
    
        # Plots
        save_loss_plot(config, model_history, save_path='loss.png', show=False)
        
        if metric_to_monitor == 'auc' or metric_to_monitor == 'val_auc':
            save_auc_plot(config, model_history, save_path='auc.png', show=False)
        if metric_to_monitor == 'accuracy' or metric_to_monitor == 'val_accuracy' or 'accuracy' in metrics:
            save_accuracy_plot(config, model_history, save_path='accuracy.png', show=False)
  
        return model, model_history
    

def load_model_o(config): 
    return tf.keras.models.load_model(config.models_path)

# Define a new Keras model class - needed to seemlessly support an old directory based format
class WrappedModel(tf.keras.Model):
    def __init__(self, loaded_model):
        super(WrappedModel, self).__init__()
        self.loaded_model = loaded_model

    def call(self, inputs):
        return self.loaded_model(inputs)
        
def load_model_from_path(model_path, verbose):
    print("model path is ", model_path)
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    try:
        # Case 1: .keras or .h5 model formats
        if model_path.endswith(".keras") or model_path.endswith(".h5"):
            print(f"Loading model from {model_path} as a Keras file.")
            model = tf.keras.models.load_model(model_path)
            print("Model Input Shape:", model.input_shape)
            print("Model Output Shape:", model.output_shape)
            return model
        # Case 2: TensorFlow SavedModel
        elif os.path.exists(model_path + "/best_model.keras"):
            new_form_model_path = model_path + "/best_model.keras"
            print(f"Loading model from {new_form_model_path} as a Keras file.")
            model = tf.keras.models.load_model(new_form_model_path)
            print("Model Input Shape:", model.input_shape)
            print("Model Output Shape:", model.output_shape)
            return model
        # Case 3: TensorFlow SavedModel
        elif os.path.isdir(model_path):
            print(f"Loading model from SavedModel directory: {model_path}")
            loaded_model = tf.saved_model.load(model_path)
            print(list(loaded_model.signatures.keys()))  
            print(type(loaded_model))
            dir(loaded_model)
            model = WrappedModel(loaded_model)
            return model
    except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            assert(True)        

def load_model(config):
    model_path = config.models_path
    return load_model_from_path(model_path, config.verbose)
