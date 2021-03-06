import os
import datetime
import utils as ut
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dense, Flatten, MaxPool2D, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def get_data(data_paths, test_size=0.2):
    '''
    Loads csv file data found in each data path and outputs the resulting training and validation images 
    and labels.

    data_paths    -> file pathd to the csv files holding the data
    test_size     -> Percentage of data set aside for validation (float between 0. and 1.)
    
    returns training and validation data (X_train, X_valid, y_train, y_valid) where:
        X_train/X_valid -> N x 3 matrices that contain image names for the center, left, and right columns
        y_train/y_valid -> N x 1 matrices that contain every steering angle
    '''
    dfs = None
    # Load each dataframe
    dfs = pd.concat([ut.load_driving_csv_df(path) for path in data_paths])
    # Specify our training images and their labels  
    X_df = dfs[['center', 'left', 'right']].values
    y_df = dfs['steering'].values
    # Split our data into training and validation sets and return the result
    return train_test_split(X_df, y_df, test_size=test_size)

def create_model(input_shape, normalization=lambda x: x, drop_rate = 0.0):
    '''
    Creates a modified version of the NVIDIA Model outlined in the post found @ https://devblogs.nvidia.com/deep-learning-self-driving-cars/

    input_shape   -> shape of the input to the model
    normalization -> function that normalizes input fed to the model
    drop_rate     -> Data to be dropped in the Dropouts used in this model
    
    returns a Keras Model
    
    NOTE:
    
    Read about dropout, variations of dropout, and various regularization techniques in the following paper: 
    https://jyx.jyu.fi/bitstream/handle/123456789/59287/URN%3ANBN%3Afi%3Ajyu-201808213890.pdf?sequence=1&isAllowed=y
    
    Read about a debate over where to put Batch Normalization: https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
    '''
        
    # Create Sequential Model
    model = Sequential()

    # Feed Input to Model and Normalize the Images
    model.add(Lambda(normalization, input_shape=input_shape))
    
    # Three convolutional layers with a 2×2 stride and a 5×5 kernel
    model.add(Conv2D(filters=24, kernel_size=5, strides=2, padding='valid', activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=2, padding='valid', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=2, padding='valid'))
    
    # Batch Normalization before RELU, followed by Dropout (both to help with regularization)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    
    # Two non-strided convolution with a 3×3 kernel size
    model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='valid'))
 
    # Batch Normalization before RELU, followed by Dropout (both to help with regularization)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    
    # Flatten
    model.add(Flatten())
    
    # Three fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10))
    
    # Batch Normalization before RELU, followed by Dropout (both to help with regularization)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    
    # Final connected layer
    model.add(Dense(1))

    return model

def train_model(model, X_train, X_valid, y_train, y_valid, batch_size=32, epochs=3, save_model_path="model.h5", pretrained_weights=None, checkpoints=False):
    '''
    Trains a model with an Adam optimizer, a Mean Squared Error loss function, and with 
    training and validation data generated by the utility image_data_batch_generator function.

    model               -> a Keras model
    X_train/X_valid     -> N x 3 matrices that contain image names for the center, left, and right columns
    y_train/y_valid     -> N x 1 matrices that contain every steering angle
    batch_size          -> how large is each batch for each epoch
    epochs              -> the number of epochs used to train this model
    save_model_path     -> string that specifies the file where the resulting model weights will be saved
    pretrained_weights  -> name of *.h5 file that holds pretrained weights for this specific model
    checkpoints         -> levarage the keras ModelCheckpoint callback class and save model after each epoch

    returns a history object of the trained model
    '''
    callbacks = []
    # finalize model and specify loss and optimizer functions
    model.compile(loss='mse', optimizer='adam')
    # initialize with pretrained weights (if there are any)
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    if checkpoints:
        callbacks += [ModelCheckpoint('model_{epoch:02d}-{val_loss:.6f}.h5', monitor='val_loss', verbose=1)]

    # create training and validation generators
    train_gen = ut.image_data_batch_generator(X=X_train, y=y_train, 
                                            batch_size=batch_size, 
                                            prepreprocessing=ut.training_preprocessing, training=True)
    valid_gen = ut.image_data_batch_generator(X=X_valid, y=y_valid, 
                                            batch_size=batch_size, 
                                            prepreprocessing=ut.validation_preprocessing)
    # train the model with the training and validation generators
    # Note: the generators should not manipulate the size of the data!
    history_object = model.fit_generator(train_gen, 
                                steps_per_epoch=len(y_train),
                                validation_data=valid_gen, 
                                validation_steps=len(y_valid), 
                                epochs=epochs,
                                verbose=1,
                                callbacks=callbacks)

    # save the model
    model.save(save_model_path)

    # return model training history
    return history_object

'''

Main

'''

# General Setup
dataPaths = ['../data', '../recovery_data']
visualizingData = True 
normalization = lambda x: x/127.5 - 1.
epochs = 3    
arch_title = '"NVIDIA Architecture"'  
changes = '""'
pretrained_weights = None # 'model_checkpoint.h5' # Set to None if not using pretrained weights
batch_size = 100

# Get Training and Validation Data
X_train, X_valid, y_train, y_valid = get_data(dataPaths)

# Visualize Data
if visualizingData:
    # Visualize Dristribution of Steering Angles as well as 
    # Before, During, and After of Several Preprocessed and Augmented Images
    ut.visualize_processed_data(count=3, X=X_train, y=y_train, save_as='images/preprocessing_visualization.jpg')

    exit()

# Create NN Model To Train On
model = create_model(input_shape=ut.NVIDIA_INPUT_SHAPE, normalization=normalization, drop_rate=0.5)

# Train Model
saveModelPath = str("model_" + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + ".h5")
history_object = train_model(model, 
                            X_train, X_valid, y_train, y_valid, 
                            save_model_path=saveModelPath,
                            epochs=epochs,
                            pretrained_weights=pretrained_weights,
                            checkpoints=True)

# Update Log
ut.update_log(history_object=history_object, 
           batch_size=batch_size,
           arch_title=arch_title,
           changes=changes)

print("Saved: ", saveModelPath)