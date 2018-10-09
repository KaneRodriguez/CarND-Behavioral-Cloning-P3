# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior (or use preexisting training data)
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[angle_distributions]: ./images/angle_distributions.jpg "angle_distributions"
[camera_views]: ./images/camera_views.jpg "camera_views"
[driving_before_and_after]: ./images/driving_before_and_after.jpg "driving_before_and_after"
[processing_stages]: ./images/processing_stages.jpg "processing_stages"
[nvidia_architecture]: ./images/nvidia_architecture.jpg "nvidia_architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

# TODO:
* Rename best model to best_model.h5
* Print model summary and save result
* Add links to code snippets referenced in this file
* Update docs on video recording
* Add kde graph to the augmentation step

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py best_model.h5
```

#### 3. Submission code is usable and readable

The code for my final solution to the problem is split amongst 2 files.

##### model.py

This file contains the code for getting training and validation data, creating the model in keras, some setup for visualizing training data, and code for training and saving the model with our data. 

##### utils.py

This utility file contains preprocessing code (`preprocess_image`, `augment_image`) and data visualization code. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I based my model off of the NVIDIA convolution neural network architecture found [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). A diagram of the NVIDIA CNN architecture is given below.

![nvidia_architecture][nvidia_architecture]

I modified this architecture to have a final 1 neuron fully connected layer at the end. Also, I added three batch normalization (BN) and dropout layers. I added one BN/dropout after the first 3 convolusions, another one after another 2 consecutive convolusions, and a final BN/dropout combination between the 3 fully connected layers and the last fully connected layer of 1 neuron.

Note: See the [Final Model Architecture](#2.-final-model-architecture) section below for further details.

#### 2. Attempts to reduce overfitting in the model

The model contains batch normalization and dropout layers in order to reduce overfitting.

Also, the provided data was split into 80% training data and 20% validation data. Image augmentation was only applied to the training data, whereas the validation data was subjected to preprocessing steps (cropping and resizing) only. Testing the efficacy of the model was done via running the model through the simulator and manually seeing if the vehicle went off the track.

#### 3. Model parameter tuning

Since the model used an adam optimizer, the learning rate did not have to be tuned manually (model.py line 25).

Also, a 50% dropout was chosen for each dropout layer. This parameter remained constant throught training.

#### 4. Appropriate training data

The training data used to keep the vehicle driving on the road was a combination of the provided Udacity data set ([link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)) and my own training data. I chose my training data based on the weak points spotted once running the models trained on the Udacity data set. I addressed these weak points by recording several passes on the sharp turn after the bridge and by recording lane recovery data (from left or right side of the road to center) all along the track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

    Modified NVIDIA Architecture w/ minimal Training Data

I initially chose to implement the NVIDIA model architecture, described [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), because this model had proven a successful real-world deep learning solution to this problem.

I adapted the NVIDIA model to this problem by adding a final 1-output fully-connected layer to the end of the architecture. This was done because I wanted only the suggested steering angle (1 output) as a result of each input image fed into the model. 

To evaluate the effectiveness of the model, the model was split into a training set and validation set. Also, I chose to use three training images that consisted of center-lane driving and recovery driving from the left and from the right side of the road.

The first training of the modified NVIDIA model had a low mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was overfitting the training data. This overfitting was likely due to there being only 3 images being used for training and validation. When running the overfitted model in the simulator, the vehicle drove straight for a few seconds before falling off to the left.

    Modified NVIDIA Architecture w/ Preprocessing and Augmentation

Since the data found in the Udacity provided data set was heavily favored towards straight driving, I decided to augment the data to rectify this uneven steering angle distribution. Techniques of random horizontal flipping, randomly choosing a camera image (center, left, or right), and randomly translating the image across the x and y axes, were inspired by [this](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project), [this](https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2), and [this](http://davidaventimiglia.com/carnd_behavioral_cloning_part1.html) student. These techniques were all made apart of the training data augmentation pipeline.

Also, I decided to preprocess the data through cropping as much of the original image as possible. This aggressive cropping was inspired by [this](http://davidaventimiglia.com/carnd_behavioral_cloning_part1.html) student. The reasoning behind aggressively cropping the image is that there is unnecessary 'future' information in each image. Given that this is a stateless nueral network, the information pertaining to the future is unnecessary and cropping that part of the image leaves more room for the network to learn pertinent features.

Training the model with the improved pipeline process resulted in a low mean squared error on the training set and relatively high mean squared error on the validation set. When testing the model on the track, the vehicle showed to be biased towards driving straight and drove straight off the track.

    Modified NVIDIA Architecture w/ Batch Normalization and Dropout

To further combat overfitting and the model's bias towards driving straight, I aimed to regularize my model by introducing batch normalization (BN) and dropout layers throughout. In the three locations that I inserted BN and dropout layers, I applied BN before a non-linearity (RELU) and dropout afterwards. 

To address the lack of lane recovery data, I manually gathered recovery focused training data from the simulator. I continually placed the vehicle on the side of the road and recorded the vehicle gradually making it's way towards the center. Also, I gathered data where the vehicle was taking sharp turns.

The result of these efforts was a slight increase in the mean squared error on the training data and a significant decrease in the mean squared error for the validation data. This model was able to drive autonomously around track one without veering off the road!

#### 2. Final Model Architecture

TODO: The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture:

![TODO][TODO]

#### 3. Creation of the Training Set & Training Process

I used a combination of the Udacity data set and lane recovery focused data that I manually collected. The distribution of the steering angle of my combined data set before and after data preprocessing and augmentation, is shown below:

![angle_distributions][angle_distributions]

One can see from the above figure that the original data set is highly concentrated in the center with most images having a steering angle close to 0. Preprocessing and augmentation causes the data to take on a 'Bell Curve' normal distribution. Although the data still appears to be centered around zero, the data is not as zero-centric as it's un-preprocessed predecessor.

Note: When initially loading in my data in a generator function, I randomly shuffle my data set.

##### Preprocessing and Augmentation

The figure below provides a visualization of several images, and their respective steering angles, as they traverse through the preprocessing and augmentation pipeline.

![processing_stages][processing_stages]

See the following sub-sections for an in-depth discussion of each image preprocessing/augmentation stage.

###### Preprocessing
 
The initial stage in preprocessing is choosing one of the available camera images. There are three viable camera orientations, and the table below shows the offset added to an image's original steering angle based on which camera orientation is chosen.

| Camera | Offset |
|:------:|:------:|
| Center | 0.00  |
| Left   | 0.26  |
| Right  | -0.26 |

A visualization of several images and their camera views is given below.

![camera_views][camera_views]

The next stage in image preprocessing consists of cropping the majority of the image and resizing the image to the input-shape specifications outlined in the NVIDIA paper discussed earlier. 

The aim of image cropping was to reduce the image down to the minimum amount necessary to make a decision. Since most of the upper portion of the image is irrelevant to the immediate action, it can be cropped. If our model was not stateless, then this extra information could have been of use.

###### Augmentation

When augmenting the data, there were several goals to keep in mind. For one, we want to fight off the highly concentrated zero-degree steering angles. Two, we want our model to better generalize in order to be robust to a plethora of varying inputs. 

Implementing a random horizontal flip to the training images allows our model to encounter a broader amount of scenarios than it would have if there was not any horizontal flipping. For example, if we have a sequence of images fed to our model that were taken at relatively close proximity, the model would not gain much from each new similar image. However, by flipping several of these images, the model is introduced to viable data from an entirely different perspective. It's like we are given free extra data!

The next data augmentation step of randomly translating the image in the x and y directions aims to distribute steering angles further from the center. By randomly translating the image in the x direction and adjusting the steering angle by 0.0035 units per pixel translated, our data set randomly shifts images in varying directions. Since the majority of our images are concentrated in the center, this (in practice) succeeds in lowering the kernel density information metric of our data.
