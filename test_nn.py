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

DEFAULT_WIDTH = 32
DEFAULT_HEIGHT = 32

def load_data(path = './data/', max_x = DEFAULT_WIDTH, max_y = DEFAULT_HEIGHT, prop = 0.2):
#         print("loading dataset")
	    
	x_train = np.empty([0, max_x, max_y, 3])
	x_test = np.empty([0, max_x, max_y, 3])

	y_train = np.empty([0])
	y_test = np.empty([0])
	label = -1

	for dirpath, dirname, filename in os.walk(path):
		x_data = []
		y_data = [] 
		for f in filename:
			fp = os.path.join(dirpath, f)	# image file
			image = imread(fp)
			#print("loading file: ", fp)
			image = cv2.resize(image, (max_y,max_x))
					
			if len(image.shape) == 3:
				# image is rgb
				x_data.append(image)
				y_data.append(label)
                
		if label != -1:
			x_data = np.array(x_data)
			y_data = np.array(y_data)
			num_of_image = x_data.shape[0]
			
			num_of_test = int(num_of_image * prop)
			num_of_train = num_of_image - num_of_test
				
			x_data_train = x_data[0:num_of_train, :]
			x_data_test = x_data[num_of_train:, :]
				
			y_data_train = y_data[0:num_of_train]
			y_data_test = y_data[num_of_train:]
			
			x_train = np.concatenate((x_train, x_data_train), axis = 0)
			x_test = np.concatenate((x_test, x_data_test), axis = 0)
			
			y_train = np.concatenate((y_train, y_data_train), axis = 0)
			y_test = np.concatenate((y_test, y_data_test), axis = 0)
				
	
		label += 1
        
	return (x_train, y_train), (x_test, y_test), label



def normalize(X_train,X_test):
	#this function normalize inputs for zero mean and unit variance
	# it is used when training a model.
	# Input: training set and test set
	# Output: normalized training set and test set according to the trianing set statistics.
	mean = np.mean(X_train,axis=(0,1,2,3))
	std = np.std(X_train, axis=(0, 1, 2, 3))
	X_train = (X_train-mean)/(std+1e-7)
	X_test = (X_test-mean)/(std+1e-7)
	return X_train, X_test

def train(model, xTrain, yTrain, xTest, yTest,
		num_classes, batchSize, max_epoches,learningRate, outFile): # best result:
	# batch_size = 80; maxepoches = 1000; learning_rate = 0.001
	# 0/1 validation loss: 0.8656
	# mess around with batch_size and maxepoches for results
	#training parameters
	
	print("batch size = ", batchSize)
	print("max epoches = ", max_epoches)
	print("learning rate = ", learningRate)
	batch_size = batchSize
	maxepoches = max_epoches
	learning_rate = learningRate
	lr_decay = 1e-6
	lr_drop = 20
	# The data, shuffled and split between train and test sets:
	# x_train: (-1, 32,32,3) numpy array
	# y_train: (-1, 1) numpy array
	# (x_train2, y_train2), (x_test2, y_test2) = cifar10.load_data()
	(x_train, y_train), (x_test, y_test) = (xTrain, yTrain),(xTest, yTest)

	# x_train = np.append(x_train2, x_train, axis = 0)
	# x_test = np.append(x_test2, x_test, axis = 0)

	# y_train = np.append(y_train2, y_train)
	# y_test = np.append(y_test2, y_test)

	print("Finished loading dataset")
	print("size of x_train = ", x_train.shape)
	print("size of x_test = ", x_test.shape)
	print("size of y_train = ", y_train.shape)
	print("size of y_test = ", y_test.shape)
	print("number of classes = ", num_classes)
	# return

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train, x_test = normalize(x_train, x_test)

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	print("y_train = ",y_train)
	print("y_test = ", y_test)

	def lr_scheduler(epoch):
		return learning_rate * (0.5 ** (epoch // lr_drop))
	reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

	#data augmentation
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False)  # randomly flip images

	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=True,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=True,  # divide each input by its std
		zca_whitening=True,  # apply ZCA whitening
		rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=True)  # randomly flip images
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)



	#optimization details
	sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


	# training process in a for loop with learning rate drop every 25 epoches.

	historytemp = model.fit_generator(datagen.flow(x_train, y_train,
									 batch_size=batch_size),
						steps_per_epoch=x_train.shape[0] // batch_size,
						epochs=maxepoches, 
						validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)
#         model.save_weights('cifar10vgg.h5')
	
	model.save_weights(outFile)
	print("Model saved as: ", outFile)

	# print("historytemp = ", historytemp)
	# plot_history(historytemp)

	return model


(x_train, y_train), (x_test, y_test), num_classes = load_data()

print("x_train = ", x_train.shape)
print("y_train = ", y_train.shape)
print("x_test = ", x_test.shape)
print("y_test = ", y_test.shape)
print("num_classes = ", num_classes)


x_shape = [DEFAULT_WIDTH,DEFAULT_HEIGHT,3]
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = x_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(16, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))
	
prediction = model.add(Dense(num_classes, activation = 'softmax'))

model.summary()

batch_size = 128 
max_epoches = 250 
learning_rate = 0.001 
outFile = "personal_train.h5"

model = train(model, x_train, y_train, x_test, y_test, num_classes,
				batchSize = batch_size, max_epoches = max_epoches,
				learningRate = learning_rate, outFile =  outFile)


predicted_x = model.predict(x_test)
residuals = np.argmax(predicted_x,1)==y_test
    
# print("predictions = ", np.argmax(predicted_x, 1))
loss = sum(residuals)/len(residuals)
print("The validation 0/1 loss is: ",loss)  
