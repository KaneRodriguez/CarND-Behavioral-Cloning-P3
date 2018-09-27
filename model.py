import os
import csv
from scipy import ndimage

trainingDataPath = "../behavioral_cloning_data/"
csvFilePath = trainingDataPath + "driving_log.csv"
imagesPath = trainingDataPath + "IMG/"

samples = []

# load in csv file with training data
with open(csvFilePath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
   
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from scipy import ndimage

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = imagesPath + batch_sample[0].split('/')[-1].split('\\')[-1]
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D

model = Sequential()

# Normalization: preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160, 320, 3),
        output_shape=(160, 320, 3)))
# Cropping
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

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

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

model.save("model.h5")
"""
Save current model characteristics to log
"""
import datetime
architecture_title = "NVIDIA Architecture"
current_notes = "First Implementation of the NVIDIA Architecture described [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars)."
fields=[str(datetime.date.today()),
        str(datetime.datetime.now().strftime('%H:%M.%f')[:-4]),
        history_object.history["loss"][-1],
        history_object.history["val_loss"][-1], 
        architecture_title,  
        len(history_object.history["loss"]), # of Epochs 
        batch_size,
        current_notes]

with open(r'log.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)