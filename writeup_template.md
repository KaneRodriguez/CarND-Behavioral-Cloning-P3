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
* Put nvidia_architecture.jpg file in images
* Print model summary and save result
* Use ann_vis library and visualize CNN
* Fill in links for students that inspired the preprocessing and data augmentation pipeline

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

I based my model off of the NVIDIA convolution neural network architecture found [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

My final solution consisted of the following architecture:

Note: See the 'Final Model Architecture' section below for further details.

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

Since the data found in the Udacity provided data set was heavily favored towards straight driving, I decided to augment the data to rectify this uneven steering angle distribution. Techniques of random horizontal flipping, randomly choosing a camera image (center, left, or right), and randomly translating the image across the x and y axes, were inspired by [this](), [this](), and [this]() student. These techniques were all made apart of the training data augmentation pipeline.

Also, I decided to preprocess the data through cropping as much of the original image as possible. This aggressive cropping was inspired by [this]() student. The reasoning behind aggressively cropping the image is that there is unnecessary 'future' information in each image. Given that this is a stateless nueral network, the information pertaining to the future is unnecessary and cropping that part of the image leaves more room for the network to learn pertinent features.

TODO: The resulting model from these pipelines wa low mean squared error on the training set and relatively high mean squared error on the validation set. When testing the model on the track

To rectify the overfitting, I decided to ... 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

![angle_distribution][angle_distribution]
![augmented_angle_distribution][augmented_angle_distribution]

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
