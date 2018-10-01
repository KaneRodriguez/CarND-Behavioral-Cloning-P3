from generators import createGenerators
from keras.preprocessing.image import ImageDataGenerator

# create training set and validation set data generators
batch_size = 32
train_generator, validation_generator, t_len, v_len = createGenerators(csvFilePath=csvFilePath, 
                                                                     imagesPath=imagesPath, 
                                                                     data_gen_pp=ImageDataGenerator(dataGenParams),
                                                                     batch_size=batch_size,
                                                                     prepreprocessing=rand_horizontal_flip)

import datetime
from model import train_model
from helper import update_log

# define file paths to relevant data
trainingDataPath = "../behavioral_cloning_data/"
csvFilePath = trainingDataPath + "driving_log.csv"
imagesPath = trainingDataPath + "IMG/"

dataGenParams = {
    'fill_mode': 'nearest', 
    'zoom_range': 0.2, 
    'rotation_range': 7, 
    'shear_range': 0.1
}

saveModelPath = str("model_" + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + ".h5")

history_object = train_model(train_generator=train_generator, train_length=t_len, 
                             validation_generator=validation_generator, validation_length=v_len, 
                             input_shape=(160, 320, 3),
                             normalization_function=lambda x: x/127.5 - 1,
                             saveModelPath=saveModelPath)

update_log(history_object=history_object, 
           batch_size=batch_size,
           arch_title='"NVIDIA Architecture"', 
           changes='"Adding more training data"')

print("Saved: ", saveModelPath)

# Note: if you get this error -> UnicodeDecodeError: 'rawunicodeescape' codec can't decode bytes in position 71-72: truncated \uXXXX
# See this stack overflow post: https://stackoverflow.com/questions/41847376/keras-model-to-json-error-rawunicodeescape-codec-cant-decode-bytes-in-posi