import os
from pathlib import Path, PureWindowsPath
import csv
import cv2
import math
import datetime
from inspect import signature
# Fixes issues when running in terminal: https://github.com/ContinuumIO/anaconda-issues/issues/1215#issuecomment-258376409
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import misc
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
    Visualize data before, during, and after preprocessing and augmentation
    
    X       -> list of center, left, and right image data
    y       -> list of angles
    count   -> how many images to visualize
    save_as -> where to save the plot
    
    does not return a value
    '''
    # Get Feasible Count
    count = min(count,len(X))
    # Save Visualizations of Camera Examples
    cam_images, cam_angles, cam_titles = [], [], []
    for cnt in range(count):
        for i in range(3):
            img, angle = rand_choose_camera(X[cnt], y[cnt], override=i)
            cam_images += [img]
            cam_angles += [angle]     
            
        cam_titles += ['Center @ ' + str(cam_angles[cnt]), 'Left @ ' + str(cam_angles[cnt + 1]), 'Right @ ' + str(cam_angles[cnt + 2])]
    plotImages(images=[ndimage.imread(path) for path in cam_images],
               titles=cam_titles,
               columns=len(cam_images)//count,
               save_as="images/camera_views.jpg")
    
    # Save visualizations of the entire pipeline (Note: Got lazy at this point and just coded what i needed so.. TODO: cleanup)
    vis_images, vis_angles, vis_titles = [], [], []
    for i in range(count):
        vis_titles += ['Chosen Camera', 'Cropped', 'Resized', 'Rand Horizontal Flip', 'Rand Translate']
        vis_img, vis_angle = rand_choose_camera(X[i], float(y[i]))
        vis_img = ndimage.imread(vis_img)
        vis_images += [vis_img]
        vis_angles += [vis_angle]
        vis_img = crop_image(vis_img)
        vis_images += [vis_img]
        vis_angles += [vis_angle]
        global NVIDIA_INPUT_SHAPE
        # Grab desired image shape
        h, w, c = NVIDIA_INPUT_SHAPE
        vis_img = resize_image(vis_img, (w, h))
        vis_images += [vis_img]
        vis_angles += [vis_angle] 
        vis_img, vis_angle = rand_horizontal_flip(vis_img, vis_angle, force=True)
        vis_images += [vis_img]
        vis_angles += [vis_angle]
        vis_img, vis_angle = rand_translate(vis_img, vis_angle)
        vis_images += [vis_img]
        vis_angles += [vis_angle]
    
    plotImages(images=vis_images, 
               titles=[str(t + ": Angle = " + format(a, '.3f')) for a, t in zip(vis_angles, vis_titles)],
               columns=len(vis_images)//count, 
               save_as="images/processing_stages.jpg")
    
    # Create Visualizations of the Entire Data Set
    y_pp = next(image_data_batch_generator(X=X, y=y, 
                                prepreprocessing=training_preprocessing,
                                training=True,
                                batch_size=len(y)))[1]
    
    save_hist([pd.Series(np.array(y)).astype(float), pd.Series(np.array(y_pp)).astype(float)],
                title='Training Set Steering Angle Distributions',  
                xlabel='Steering Angle', 
                ylabel='Count', 
                key=['Original','Augmented'],
                bins=30, 
                save_as= 'images/angle_distributions.jpg',
                xlim=[-0.75, 0.75],
                ylim=[0, 2500])
    

def save_hist(dfs, title, xlabel, ylabel, bins, key, save_as, xlim=None, ylim=None):
    '''
    Creates a chart from each DataFrame in 'dfs' and saves the resulting figure
l
    dfs                   -> array of pandas DataFrames or Series objects
    title, xlabel, ylabel -> plot title, x, and y labels
    bins                  -> the number of bins in which to group the data
    key                   -> array of labels for the plot legend
    save_as               -> path to where the figure image will be saved
    xlim, ylim            -> x and y axes limits (see matplotlib.pyplot doc for these functions)
    
    does not return any value
    '''
    fig, ax = plt.subplots()
    
    plt.hist(dfs, bins=bins, rwidth=0.85, alpha=0.8, label=key)
   
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
        
    ax.legend(loc='upper right') 
    plt.title(title)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.9)

#     for i, df in enumerate(dfs):
#         print(i)
#         ax2 = df.plot.kde(ax=ax, secondary_y=True, legend=True, label=key[i] + ' KDE')
#         ax2.set_ylabel('Density')
 
    plt.savefig(save_as)
    plt.gcf().clear()

def rand_choose_camera(cameras, angle, override=None):
    '''
    Randomly choose the center, left, or right camera.
    
    cameras     -> List of Center, Left, and Right camera images/filepaths
    angle       -> the angle for the center camera
    override    -> number to override any randomness. I.e. - 0 (Center), 1 (Left), or 2 (Right)
    
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
    # Apply override if given
    if override is not None:
        choice = override
    return cameras[choice], cam_switch[choice](float(angle))
    
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

def preprocess_image(img):
    '''
    Applies preprocessing techniques to the input img.

    img       -> a 160 x 320 x 3 RGB image
    
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

def rand_horizontal_flip(img, angle, force=False):
    '''
    Randomly flips the input image and reverses angle direction.

    img   -> an image of any dimensions
    angle -> steering angle
    force -> forces a flip
    
    returns a tuple of (img, angle)
    '''
    if force or np.random.rand() < 0.5:
        # flip the image horizontally
        img = cv2.flip(img, 1)
        # reverse the sign of the steering angle
        angle *= -1
    
    return img, angle

def validation_preprocessing(x, y):
    '''
    Applies normal preprocessing to the image. Meant for when 
    applying preprocessing to data from a validation set.

    x           -> image
    y           -> label (steering angle)

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