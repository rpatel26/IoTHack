
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

LOAD_PATH = "./personal_train.h5"
IMG_SRC_DIR = "./testing_data/"

DEFAULT_WIDTH = 32
DEFAULT_HEIGHT = 32

model = Sequential()

model.load_weights(LOAD_PATH)

while True:
    cpt = sum([len(files) for r, d, files in os.walk(IMG_SRC_DIR)])
    if(cpt == 1):
        file_list = glob.glob(os.path.join(IMG_SRC_DIR,'*'))
        
        image = cv2.imread(file_list[0])
            
        if image is None:
            # print("Image is of type None")
            continue
    
    print("File detected!!")
    print(file_list)
        
    image = cv2.resize(image, (DEFAULT_WIDTH,DEFAULT_HEIGHT))
    image = np.expand_dims(image, axis = 0)
            
    predicted_values = model.predict(image) # sum of every element adds up to 1
    result = classes[np.argmax(predicted_values, axis = 1)[0] + 1]
    
    print("result = ",result)
    print("count = ", count)
    fp = open(RESULT_DEST_DIR, "w")
    
    fp.write("Result: ")
    fp.write(result)
    fp.write("\n")

    '''
        fp.write(str(count))
        fp.write("\n")
        '''
    
    fp.close()
    count = count + 1
    
    #time.sleep(5)
    os.remove(file_list[0])
# os.remove(RESULT_DEST_DIR)


