import numpy as np
import random
import os
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, ConvLSTM2D, Dropout, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import ModelCheckpoint


'''The model works with video chunks of recorded speckle pappern'''

def set_seed(seed_for_init = 1, random_seed = 9):
    np.random.seed(seed_for_init)  # Set seed for NumPy operations to ensure reproducibility
    random.seed(random_seed)
    
    ''' specific for Tensorflow: '''

    # Ensure deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    tf.random.set_seed(seed_for_init)  # Set seed for TensorFlow operations to ensure reproducibility


def define_model(config, sz_conv, sz_dense):
    """Defines a ConvLSTM2D model for classification tasks."""
    model = Sequential([
        BatchNormalization(),
        ConvLSTM2D(
            filters=sz_conv,
            kernel_size=(3, 3),
            kernel_constraint=max_norm(2.),
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=0),
            return_sequences=False,
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
        BatchNormalization(),
        Dropout(0.5)
    ])

    # Add output layer and compile the model based on number of classes
    output_units = config.number_of_classes if config.number_of_classes > 2 else config.number_of_classes - 1
    activation = "softmax" if config.number_of_classes > 2 else "sigmoid"
    loss = 'categorical_crossentropy' if config.number_of_classes > 2 else 'binary_crossentropy'

    model.add(Dense(output_units, kernel_constraint=max_norm(2.), activation=activation))
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    return model

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
  
def train_model(config, sz_conv, sz_dense, x_train, y_train, x_val, y_val, batch_sz, n_epochs):
    """Trains the model and saves the best model based on validation accuracy."""
    model_checkpoint_callback = ModelCheckpoint(
        filepath=config.models_path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    model = define_model(config, sz_conv, sz_dense)
    model_history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_sz,
        epochs=n_epochs,
        callbacks=[model_checkpoint_callback]
    )
    
    save_accuracy_plot(config, model_history, save_path = 'accuracy.png', show = False)
    save_loss_plot(config, model_history, save_path = 'loss.png', show = False)
    
    return model, model_history

def load_model(config): 
    return tf.keras.models.load_model(config.models_path)
