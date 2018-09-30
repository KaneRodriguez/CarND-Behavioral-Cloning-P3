# Create Architecture using Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Lambda, Cropping2D

def train_model(train_generator, train_length, validation_generator, validation_length, input_shape, input_lambda, saveModelPath='model.h5'):
    # Create Sequential Model
    model = Sequential()

    # Feed Input to Model
    model.add(Lambda(lambda x: input_lambda(x), input_shape=input_shape))
    
    # Three convolutional layers with a 2×2 stride and a 5×5 kernel
    model.add(Conv2D(filters=24, kernel_size=5, strides=2, padding='valid', activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=2, padding='valid', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=2, padding='valid', activation='relu'))

    # Two non-strided convolution with a 3×3 kernel size
    model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))

    # Flatten
    model.add(Flatten())

    # Three fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10))
    # Final connected layer
    model.add(Dense(1))

    # finalize model and specify loss and optimizer functions
    model.compile(loss='mse', optimizer='adam')
    # train model with generators
    history_object = model.fit_generator(train_generator, 
                                        steps_per_epoch=train_length,
                                        validation_data=validation_generator, 
                                        validation_steps=validation_length, 
                                        epochs=3,
                                        verbose=1)

    # save the model
    model.save(saveModelPath)

    # return model training history
    return history_object