import os
import csv
import datetime
import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# from statistics import median

DATA_PATH = "../behavioral_cloning_data/"

def read_image(fileName):
    '''
    Reads in an RGB? image

    fileName -> relative or absolute file path to the image

    returns a 3 Channel RGB image
    '''
    return ndimage.imread(fileName)

def image_data_batch_generator(X, y, batch_size=32, data_gen=ImageDataGenerator(), prepreprocessing=lambda x, y: (x, y)):
    '''
    Creates a generator that applies an ImageDataGenerator and custom prepreprocessing logic 
    while loading training images and labels.

    X                -> n x 3 matrix w/ center, left, and right image names
    y                -> n x 1 matrix w/ steering data
    batch_size       -> how large is each batch
    data_gen         -> image preprocessing generator to be used
    prepreprocessing -> function to apply to batch images before they are fed to the image preprocessing generator
    
    returns a generator that returns batched, loaded, and preprocessed X and y data

    NOTE:

    See this blog post for more info on creating generators for the 'fit_generator' Keras function:
    https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
    
    '''    
    # use the globally defined path to the data to define our IMG directoy
    global DATA_PATH
    img_dir = DATA_PATH + "IMG/"

    # generators loop forever
    while 1:
        # run through each batch of X & y
        for offset in range(0, len(X), batch_size):
            # assign the current batch
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]
            # initialize batch preprocessed data containers
            images = []
            angles = []
            # loop through each batch
            for image_names, angle in zip(batch_X, batch_y):
                # TODO: method for using all of images OR for choosing amongst them and 
                #       updating their respective label accordingly
                # Use the first image (center) for now
                center_image = read_image(img_dir + image_names[0].split('/')[-1].split('\\')[-1])
                center_angle = float(angle)
                # apply any pre-preprocessing necessary
                center_image, center_angle = prepreprocessing(center_image, center_angle)
                # add image and angle to current batch containers
                images.append(center_image)
                angles.append(center_angle)

            # convert lists to numpy arrays
            images = np.array(images)
            angles = np.array(angles)
            # fit the preprocessing generator to images (TODO: might not be needed)
            data_gen.fit(images)
            # preprocess the images and return the result (this for loop runs once due to batch_size being the current batches size)
            for x_batch, y_batch in data_gen.flow(x=images, y=angles, batch_size=len(images), shuffle=True):
                yield x_batch, y_batch

'''
Helper for logging information on previous model training session
'''

def update_log(history_object, batch_size, logPath=r'log.csv', arch_title='Architecture', changes='""'):
    fields=[str(datetime.date.today()),
            str(datetime.datetime.now().strftime('%H:%M')),
            history_object.history["loss"][-1],
            history_object.history["val_loss"][-1], 
            arch_title,  
            len(history_object.history["loss"]), # of Epochs 
            batch_size,
            changes,
            '""']

    # append new log data to end of log file
    with open(logPath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)