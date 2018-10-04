import os
import csv
import cv2
import math
import datetime
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# from statistics import median

# Define Globals
DATA_PATH = "../behavioral_cloning_data/"
NVIDIA_INPUT_SHAPE = (66, 200, 3)

def save_hist(df, title, xlabel, ylabel, bins, save_as):
    '''
    Creates a histogram and saves the resulting figure

    df                    -> pandas DataFrame or Series object
    title, xlabel, ylabel -> plot title, x, and y labels
    bins                  -> the number of bins in which to group the data
    save_as               -> path to where the figure image will be saved
    
    does not return any value
    '''
    df.plot.hist(grid=True, bins=bins, rwidth=0.85, color='dodgerblue')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.8)

    plt.savefig(save_as)

def crop_image(img):
    '''
    Crops the image to cut off the bottom car hood and top sky

    img -> 160 x 320 x 3 RGB image

    returns a ? x 320 x 3 RGB image

    NOTE:

    Check out this sweeeet article: http://davidaventimiglia.com/carnd_behavioral_cloning_part1.html
    '''
    return img[81:-22, :,:]

def resize_image(img, dsize):
    '''
    Resizes the image to the dsize dimensions

    img   -> input image
    dsize -> (height, width) tuple specifying output image size

    returns img with dsize dimensions (channels unchanged)
    '''
    return cv2.resize(img, dsize)

def preprocess_image(img):
    '''
    Applies preprocessing techniques to the input img.

    img      -> a 160 x 320 x 3 RGB image

    returns a 66 x 200 x 3 RGB image

    NOTE: 
    Applied to images used to train the model AND to images
    fed to the model for prediction. 
    '''
    global NVIDIA_INPUT_SHAPE

    # Grab desired image shape
    h, w, c = NVIDIA_INPUT_SHAPE
    # Crop the image
    img = crop_image(img)
    # Resize the image back to it's original dimensions
    img = resize_image(img, (w,h))
    # Convert to YUV?
    # Other?

    return img

def augment_image(img, angle):
    '''
    Augments the given image and angle in order to achieve close to zero variance in steering angle distribution.

    img   -> input image
    angle -> steering angle

    returns the augmented image and angle
    '''
    # Randomly flip the image horizontally
    img, angle = rand_horizontal_flip(img, angle)

    return img, angle 

def rand_horizontal_flip(img, angle):
    '''
    Randomly flips the input image and reverses angle direction.

    img -> an image of any dimensions

    returns a tuple of (img, angle)
    '''
    if np.random.rand() < 0.5:
        # flip the image horizontally
        cv2.flip(img, 1)
        # reverse the sign of the steering angle
        angle *= -1
    
    return img, angle

def filename_from_path(path):
    '''
    Gets filename from path

    path -> path to the file (including the file name and extension)

    returns the file name (including the extension)
    '''
    return path.split('/')[-1].split('\\')[-1]

def read_image(fileName):
    '''
    Reads in an RGB? image

    fileName -> name and extension of the image file (does not include any path)

    returns a 3 Channel RGB image
    
    NOTE:

    The path to the file itself is a global variable
    '''
    # use the globally defined path to the data to define our IMG directoy
    global DATA_PATH
    img_dir = DATA_PATH + "IMG/"

    return ndimage.imread(img_dir + fileName)

def validation_preprocessing(x, y):
    '''
    Applies normal preprocessing to the image. Meant for when 
    applying preprocessing to data from a validation set.

    x -> image
    y -> label (steering angle)

    returns a tuple (x, y)

    NOTE:

    For now, this is simply a wrapper to the preprocess_image function
    '''
    return preprocess_image(x), y

def training_preprocessing(x, y):
    '''
    Intended for model training images only. Applies augmentation and image preprocessing to the
    image and it's label.

    x -> image
    y -> label (steering angle in this case)

    returns a tuple (x, y) where x is the same shape as the input imag and y is the angle label
    '''
    # Augment the data then apply prediction preprocessing to the data
    return validation_preprocessing(*augment_image(x,y))

def image_data_batch_generator(X, y, batch_size=32, data_gen=ImageDataGenerator(), prepreprocessing=lambda x,y: (x,y)):
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
                center_image = read_image(filename_from_path(image_names[0]))
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
    fields=[str(datetime.date.today()), # Today's Date
            str(datetime.datetime.now().strftime('%H:%M')), # Today's Hour and Minute
            history_object.history["loss"][-1], # The final mse loss on the training set
            history_object.history["val_loss"][-1],  # The final mse loss on the validation set
            arch_title, # The model architecture being used
            len(history_object.history["loss"]), # The number of Epochs 
            batch_size,
            changes, # Any notable changes to the model after last save
            '""'] # Space for any notes about model performance in the simulator

    # append new log data to end of log file
    with open(logPath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

'''
    Helper for plotting multiple images and giving them titles
'''

def plotImages(images, titles=[""], columns=1, figsize=(20,10), gray=False, save_as='', row_titles=False):
    errorStr = "plotImages failed..."
    # images and titles must be lists
    if(not isinstance(images, (list,)) or not isinstance(titles, (list,))):
        print(errorStr + " images/titles are not both instances of list")
        return
    
    # the number of titles must match the number of columns if not in row_title mode OR
    # the number of titles must match the number of images
    if((not row_titles and len(titles) != columns) and len(titles) != len(images)):
        print(errorStr + " images/titles are not the same length")
        return
    
    rows = math.ceil(len(images) / columns)

    # the number of titles must match the number of rows if in row_titles mode
    if(len(titles) != rows and row_titles):
        print(errorStr + " in row_titles mode and number of titles does not match the number of rows")
        return

    plt.figure(figsize=figsize)
    
    fig = plt.gcf()
    
    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        
        if len(images) == len(titles):
            title = titles[i]
        elif not row_titles:
            title = titles[i % columns]
        else:
            title = titles[i % rows]

        plt.gca().set_title(title)
       
        # if gray is a list, each item  
        # corresponds to if each row is gray
        tmpGray = gray
        if isinstance(gray, (list,)):
            tmpGray = gray[i // columns]
            
        if gray:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

    if save_as != '':
        fig.savefig(save_as, dpi=fig.dpi)