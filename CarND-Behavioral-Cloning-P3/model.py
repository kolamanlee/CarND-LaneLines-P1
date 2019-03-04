
#import the lib
import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from sklearn.model_selection import train_test_split
import sklearn
###
###


#read the csv contents  to a list 
def read_csv(path):
	csv_path = os.path.join(path, 'driving_log.csv')
	lines = []

	with open(csv_path) as csv_file:
		reader =csv.reader(csv_file)
		for line in reader:
			lines.append(line)
	return lines


#create a generator (using yield feature) to read the data when need it during the traing model.
def read_generator(root_dir, samples, batch_size=64):
	num_samples = len(samples)
	step_len = batch_size//2
	while(1):## loop never stop
		sklearn.utils.shuffle(samples)
		for offset in range(1, num_samples, step_len):
			batch_samples = samples[offset:offset+step_len]
			images = []
			steerings = []
			for one_line_sample in batch_samples:
				img_path = os.path.join(root_dir, one_line_sample[0])
				center_image = cv2.imread(img_path)
				if center_image is not None:
					center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
					steering = float(one_line_sample[3])
					images.append(center_image)
					steerings.append(steering)

					#add augmented image and angle
					images.append(cv2.flip(center_image, 1))
					steerings.append(-steering)
				else:
					print('None img:',img_path)

			X_train = np.array(images)
			y_train = np.array(steerings)
			#print('X_train.shape',X_train.shape)

			yield (X_train, y_train)


## a test networks for a trail
def test_network():
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
	model.add(Flatten())
	model.add(Dense(1))
	return model

## create the Nvidia Network
def nvidia_network():
	model = Sequential()
	# pre-process the data...
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
	model.add(Cropping2D(cropping = ((50,20),(0,0))))
	#
	model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = "relu"))
	model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = "relu"))
	model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = "relu"))
	model.add(Convolution2D(64, 3, 3, activation = "relu"))
	model.add(Convolution2D(64, 3, 3, activation = "relu"))
	model.add(Flatten())
	model.add(Dropout(0.3))	
	model.add(Dense(100))
	model.add(Dropout(0.3))
	model.add(Dense(10))
	model.add(Dense(1))

	##
	return model

def pipeline():
		# read the training data from the folders
		root_dir = './data/'
		samples = read_csv(root_dir)

		print('len(samples)=', len(samples))
		#

		# Splitting samples.
		train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

		# creating generators
		train_generator = read_generator(root_dir, train_samples, batch_size = 32)
		validation_generator = read_generator(root_dir, validation_samples, batch_size = 32)
		#Model creation
		model = nvidia_network()

		model.compile(loss = 'mse', optimizer = 'adam')
		history_object = model.fit_generator(train_generator, \
							samples_per_epoch = len(train_samples),\
							validation_data = validation_generator, \
							nb_val_samples = len(validation_samples),\
							nb_epoch = 3,\
							verbose = 1)
		model.save('model.h5')

		### print the keys contained in the history object
		print(history_object.history.keys())

		### plot the training and validation loss for each epoch
		plt.plot(history_object.history['loss'])
		plt.plot(history_object.history['val_loss'])
		plt.title('model mean squared error loss')
		plt.ylabel('mean squared error loss')
		plt.xlabel('epoch')
		plt.legend(['training set', 'validation set'], loc='upper right')
		plt.show()
        


print('start ...')

pipeline()

print('finish ...')

