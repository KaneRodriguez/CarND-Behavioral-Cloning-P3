import os
from pathlib import Path, PureWindowsPath
import csv
import cv2
import math
import datetime
# Fixes issues when running in terminal: https://github.com/ContinuumIO/anaconda-issues/issues/1215#issuecomment-258376409
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import pandas as pd
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Define Globals
NVIDIA_INPUT_SHAPE = (66, 200, 3)

def load_driving_csv_df(dataPath):
    ''' 
    Loads in a 'driving_log.csv' file found at the given file path 
    into a pandas dataframe and returns the result
    
    dataPath  -> file path to the csv file
    
    returns a pandas data frame
    
    NOTE:
    
    The csv files should contain data in the following order:
    
    Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, and Speed
    0,            1,          2,           3,              4,        5,         6
    '''
    # driving data expected in 'driving_log.csv' files
    csv_file_path = os.path.join(dataPath, 'driving_log.csv')
    # Load in file with pandas
    df = pd.read_csv(csv_file_path, ",",
                     names=["center", "left", "right", "steering", "throttle", "break", "speed"])
    # drop first row if it is a header
    if len(df.index) > 0:
        try:
            float(df.loc[df.index[0], 'steering'])
        except ValueError:
            df = df[1:]
            
    # strip out all but image name and replace with '../DATA_PATH/IMG/IMAGE_NAME.jpg'
    strip = lambda x: (Path(dataPath) / Path('IMG') / PureWindowsPath(x).name)
    df['center'] = df['center'].map(strip)
    df['left'] = df['left'].map(strip)
    df['right'] = df['right'].map(strip)
    
    return df
                
def visualize_processed_data(X, y, count, save_as):
    '''
    Visualize data before and after preprocessing and augmentation
    
    X       -> list of center, left, and right image data
    y       -> list of angles
    count   -> how many images to visualize
    save_as -> where to save the plot
    
    does not return a value
    '''
    # Get Feasible Count
    count = min(count,len(X))
    # Prepare Dict to Store Visualization Data in
    vis_data = {
        'images':{
            'orig':[],
            'pp':[],
            'aug':[]
        }, 
        'angles':{
            'orig':[],
            'pp':[],
            'aug':[]
        }
     }
    # Loop through data and preprocess, augment, and store the results
    for i, (paths, angle) in enumerate(zip(X[:count], y[:count])):
        image_path, angle = rand_choose_camera(paths, float(angle))
        img = ndimage.imread(image_path)
        # store original
        vis_data['images']['orig'].append(img)
        vis_data['angles']['orig'].append(angle)
        # store preprocessed                           
        tmpImg, tmpAng = validation_preprocessing(np.copy(img), angle)
        vis_data['images']['pp'].append(tmpImg)
        vis_data['angles']['pp'].append(tmpAng)
        # store preprocessed + augmented 
        tmpImg, tmpAng = augment_image(np.copy(tmpImg), tmpAng)
        vis_data['images']['aug'].append(tmpImg)
        vis_data['angles']['aug'].append(tmpAng)   
        
    # Combine all visualization images and angles
    vis_images = vis_data['images']['orig'] + vis_data['images']['pp'] + vis_data['images']['aug']
    vis_angles = vis_data['angles']['orig'] + vis_data['angles']['pp'] + vis_data['angles']['aug']
    
    # Save Visualizations
    plotImages(images=vis_images, 
                titles=vis_angles, 
                columns=count, 
                save_as=save_as)

def save_kde(dfs, title, xlabel, ylabel, bins, key, save_as):
    '''
    Creates a kde chart from each DataFrame in 'dfs' and saves the resulting figure
l
    dfs                   -> array of pandas DataFrames or Series objects
    title, xlabel, ylabel -> plot title, x, and y labels
    bins                  -> the number of bins in which to group the data
    key                   -> array of labels for the plot legend
    save_as               -> path to where the figure image will be saved
    
    does not return any value
    '''
    fig, ax = plt.subplots()
    
    #plt.hist(dfs, bins=bins, rwidth=0.85, alpha=0.8, label=key)
   
    for i, df in enumerate(dfs):
      df.plot.kde(ax=ax, secondary_y=False, label=key[i])
       
    ax.legend(loc='upper right') 
    plt.title(title)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.9)

    plt.savefig(save_as)
    plt.gcf().clear()

def rand_choose_camera(cameras, angle):
    '''
    Randomly choose the center, left, or right camera.
    
    cameras -> List of Center, Left, and Right camera images/filepaths
    angle   -> the angle for the center camera
    
    returns a tuple of (camera, angle)
    
    NOTE:
    The data in cameras is not accessed, so passing in either file path or image works
    
    Also, the adjustment angle was based on: https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project
    '''
    # Choose a random camera
    choice = np.random.choice(3, 1)[0]
    # Mutate angle based on which camera
    offset = 0.26
    cam_switch = {
        0: lambda a: a, # Center
        1: lambda a: a + offset, # Left
        2: lambda a: a - offset # Right
    }
    return cameras[choice], cam_switch[choice](angle)
    
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

def rand_translate(img, angle):
    '''
    Translates an image by a random x and random y
    
    img   -> input image
    angle -> steering angle

    NOTE:
    Chosen angle to pixel ratio chosen from here: https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2
    '''
    # Grab image dimensions
    h, w, c = img.shape
    # Define Angle Change per Pixel 
    angle_per_x_pixel = 0.0035
    # shift x and y by a random percentage 
    # of a fraction of the width and height
    x_shift = (np.random.rand()*2.-1.)*(w/7.)
    y_shift = (np.random.rand()*2.-1.)*(h/22.)
    # create transformation matrix
    M = np.float32([ [1,0,x_shift], # x-shift
                     [0,1,y_shift]  # y-shift
                   ])
    # apply transformation matrix to the full image
    img = cv2.warpAffine(img, M, (w, h))
    # adjust angle based on number of shifted x pixels
    angle += x_shift*angle_per_x_pixel
    
    return img, angle 

def visualize_training_pipeline(X, y, count=1):
    '''
    Visualize Training Data Distribution Before and After Augmentation, as well
    as visualizations of each stage of the pipeline.
    '''
    # Pandas Histogram Plotting Tutorial: https://realpython.com/python-histograms/
    y_pp = next(image_data_batch_generator(X=X, y=y, 
                                prepreprocessing=training_preprocessing,
                                training=True,
                                batch_size=len(y)))[1]
    
    save_kde([pd.Series(np.array(y)).astype(float), pd.Series(np.array(y_pp)).astype(float)],
                title='Training Set Steering Angle KDE Distributions',  
                xlabel='Steering Angle', 
                ylabel='Density', 
                key=['Original','Augmented'],
                bins=20, 
                save_as= 'images/angle_distributions.jpg' )
    # Follow one image through it's pipeline
    funcs = []
    save_images = [f(img) for f in funcs]
    save_titles = []

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
    # Resize image to the desired model input shape
    img = resize_image(img, (w,h))
    # Convert to YUV?
    # Other?

    return img

def augment_image(img, angle):
    '''
    Augments the given image and angle in order to help achieve 
    close to zero variance in steering angle distribution.

    img   -> input image
    angle -> steering angle

    returns the augmented image and angle
    '''
    # Randomly flip the image horizontally
    img, angle = rand_horizontal_flip(img, angle)
    # Randomly translate the image
    img, angle = rand_translate(img, angle)
    # Brightness?
    # Rotation?
    # Shadow?
    # Other?
    return img, angle 

def rand_horizontal_flip(img, angle):
    '''
    Randomly flips the input image and reverses angle direction.

    img   -> an image of any dimensions
    angle -> steering angle
    
    returns a tuple of (img, angle)
    '''
    if np.random.rand() < 0.5:
        # flip the image horizontally
        img = cv2.flip(img, 1)
        # reverse the sign of the steering angle
        angle *= -1
    
    return img, angle

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
    # Apply preprocessing to the data then augment the data
    return augment_image(*validation_preprocessing(x, y))

def image_data_batch_generator(X, y, batch_size=32, prepreprocessing=lambda x,y: (x,y), training=False):
    '''
    Creates a generator that applies an ImageDataGenerator and custom prepreprocessing logic 
    while loading training images and labels.

    X                -> n x 3 matrix w/ center, left, and right image names
    y                -> n x 1 matrix w/ steering data
    batch_size       -> how large is each batch
    prepreprocessing -> function to apply to batch images
    training         -> speficies if this is training data
    
    returns a generator that returns batched, loaded, and preprocessed X and y data

    NOTE:

    When 'training' is set, the center image will be chosen for all X batch data

    Also, see this blog post for more info on creating generators for the 'fit_generator' Keras function:
    https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
    '''    
    # Shuffle the data
    X, y = shuffle(X, y)
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
                # Make angle into a float
                angle = float(angle)
                # Randomly choose an image to load if training or load center image if not training
                image_path, cur_angle = (rand_choose_camera(image_names, angle), (image_names[0], angle))[training]
                # Load Image 
                loaded_image = ndimage.imread(image_path)
                # apply any pre-preprocessing necessary
                loaded_image, cur_angle = prepreprocessing(loaded_image, cur_angle)
                # add image and angle to current batch containers
                images.append(loaded_image)
                angles.append(cur_angle)

            # convert lists to numpy arrays and return the result
            yield np.array(images), np.array(angles)

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
            title = titles[i//columns]

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