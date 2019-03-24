
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import cv2
from scipy import misc
from scipy.ndimage import imread
import os
import argparse

LOAD_PATH = 'personal_train.h5'
IMG_SRC = 'testing_data/e5f38070-a16c-4a09-8345-a6097f6a9e11.JPG'
dataset_path = './data'

DEFAULT_WIDTH = 32
DEFAULT_HEIGHT = 32

# retun classification at index + 1
classes = [x[0] for x in os.walk(dataset_path)]
num_classes = len(classes) - 1

x_shape = [DEFAULT_WIDTH,DEFAULT_HEIGHT,3]
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = x_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(16, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

'''
model.add(Conv2D(8, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
'''

model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))

num_classes = 2
prediction = model.add(Dense(num_classes, activation = 'softmax'))

model.summary()

model.load_weights(LOAD_PATH)

def classify_data(data):
	image = cv2.resize(data, (DEFAULT_WIDTH,DEFAULT_HEIGHT))
	image = np.expand_dims(image, axis = 0)
            
	predicted_values = model.predict(image) # sum of every element adds up to 1
	result = classes[np.argmax(predicted_values, axis = 1)[0] + 1]
    
	print("result = ",result)
	print("predicted values = ", predicted_values)

'''
image = cv2.imread(IMG_SRC)
classify_data(image)
'''
