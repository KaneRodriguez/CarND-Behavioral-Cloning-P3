import os
import csv
import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# define file paths to relevant data
trainingDataPath = "../behavioral_cloning_data/"
csvFilePath = trainingDataPath + "driving_log.csv"
imagesPath = trainingDataPath + "IMG/"

# load in csv file with training data
csvLines = []
with open(csvFilePath) as dataCsvFile:
    csvLines = [line for line in csv.reader(dataCsvFile)]

# split data into training and validation sets
train_lines, validation_lines = train_test_split(csvLines, test_size=0.2)

# define data generator
def generator(lines, pathToImages, batch_size=32, normalization_preprocessing_only=False):
    # define image preprocessing generator
    image_preprocessing_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    # check if more preprocessing required
    if normalization_preprocessing_only is False:
        # add extra preprocessing
        image_preprocessing_generator = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            vertical_flip=False)
            
    # generators loop forever
    while 1:
        # run through each batch of lines
        for offset in range(0, len(lines), batch_size):
            # assign the current batch lines
            batch_lines = lines[offset:offset+batch_size]
            # initialize batch preprocessed data containers
            images = []
            angles = []
            # loop through each batch line
            for line in batch_lines:
                # assign some content based on data in that line
                name = pathToImages + line[0].split('/')[-1].split('\\')[-1]
                center_image = ndimage.imread(name)
                center_angle = float(line[3])
                # add image and angle to current batch containers
                images.append(center_image)
                angles.append(center_angle)
        
            # convert lists to numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            # fit the image preprocessing generator to images
            image_preprocessing_generator.fit(X_train)
            # preprocess the images and return the result (this for loop runs once due to batch_size being the current batch size)
            for x_batch, y_batch in image_preprocessing_generator.flow(x=X_train, y=y_train, batch_size=len(batch_lines), shuffle=True):
                yield x_batch, y_batch

# create training and validation generators
batch_size = 32
train_generator = generator(train_lines, imagesPath, batch_size=batch_size)
validation_generator = generator(train_lines, imagesPath, batch_size=batch_size, normalization_preprocessing_only=True)

# Create Architecture using Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Lambda, Cropping2D

# Create Sequential Model
model = Sequential()

# Feed Input to Model and Crop
model.add(Cropping2D(input_shape=(160, 320, 3), cropping=((70, 25), (0, 0))))

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
history_object = model.fit_generator(train_generator, 
                                    steps_per_epoch=len(train_lines),
                                    validation_data=validation_generator, 
                                    validation_steps=len(validation_lines), 
                                    epochs=5,
                                    verbose=1)

model.save("model.h5")
"""
Save current model characteristics to log
"""
import datetime
architecture_title = '"NVIDIA Architecture"'
notable_changes = '"Captured data of vehicle driving from each side of road to center and of vehicle driving in lane center."'
fields=[str(datetime.date.today()),
        str(datetime.datetime.now().strftime('%H:%M')),
        history_object.history["loss"][-1],
        history_object.history["val_loss"][-1], 
        architecture_title,  
        len(history_object.history["loss"]), # of Epochs 
        batch_size,
        notable_changes,
        '""']

with open(r'log.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)