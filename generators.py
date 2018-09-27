import os
import csv
import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# from statistics import median

def createGenerators(csvFilePath, imagesPath, batch_size=32, test_size=0.2, data_gen_pp=ImageDataGenerator(), subsegment=None, ignoreFirst=False):
    '''
        csvFilePath  -> relative path to the driving_log.csv file that holds all relevant data (TODO: perhaps cast to absolute file path before passing into this function)
        batch_size   -> how large is each batch
        test_size    -> what percentage is allocated towards testing (float between 0. and 1.)
        data_gen_pp  -> image preprocessing generator to be used on the training set only
        subsegment   -> only process part of the csv file (DEBUG ONLY)
    '''    
    csvLines = []
    # load in csv file with training data
    with open(csvFilePath) as dataCsvFile:
        # read in each line
        csvLines = [line for line in csv.reader(dataCsvFile)]
    
        # truncate the first line if flag set and len > 0
        if ignoreFirst and len(csvLines) > 0:
            csvLines = csvLines[1:len(csvLines)]
    
        if subsegment is not None: 
            csvLines = csvLines[0:min(len(csvLines)-1, subsegment)]
            # DEBUGGING ONLY. Only use the leftest, rightest, and centerest images
            # fr = subsegment
            # csvLines = np.array(sorted(csvLines, key=lambda x : float(x[3])))
            # csvLines = np.array(sorted(csvLines, key=lambda x : float(x[3])))
            # lgt = len(csvLines)
            # csvLines = csvLines[np.r_[0:lgt//fr, lgt//2 - lgt//fr:lgt//2 + lgt//fr, lgt//fr*(fr - 1):lgt]]

    # split data into training and validation sets
    train_lines, validation_lines = train_test_split(csvLines, test_size=0.2)
    
    # define data generator
    def generator(lines, pathToImages, batch_size=32, data_gen=ImageDataGenerator()):
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
                # fit the preprocessing generator to images (TODO: might not be needed)
                data_gen.fit(X_train)
                # preprocess the images and return the result (this for loop runs once due to batch_size being the current batch size)
                for x_batch, y_batch in data_gen.flow(x=X_train, y=y_train, batch_size=len(batch_lines), shuffle=True):
                    yield x_batch, y_batch

    # create training and validation generators
    train_generator      = generator(train_lines, imagesPath, batch_size=batch_size, data_gen=data_gen_pp)
    validation_generator = generator(validation_lines, imagesPath, batch_size=batch_size)

    # return both generators and the lengths of the respective data sets that they iterate over
    return train_generator, validation_generator, len(train_lines), len(validation_lines)