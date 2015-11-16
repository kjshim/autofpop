from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from autofpop.new_recognizer import RecognizerCommon

class RecognizerDL(RecognizerCommon):
	def fit(self):
		# X = np.array(self.data.X)
		X = self.data.X

		dimof_input = len(self.data.X[0])
		dimof_output = len(set(self.data.y))
		print('dimof_input: ', dimof_input)
		print('dimof_output: ', dimof_output)

		inverse, y = np.unique(self.data.y, return_inverse=True)
		y = np_utils.to_categorical(y, dimof_output)

		batch_size = 128
		dimof_middle = 1000
		dropout = 0.3
		countof_epoch = 20 #100
		verbose = 1 #0
		print('batch_size: ', batch_size)
		print('dimof_middle: ', dimof_middle)
		print('dropout: ', dropout)
		print('countof_epoch: ', countof_epoch)
		print('verbose: ', verbose)
		print()

		model = Sequential()
		# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
		# this applies 32 convolution filters of size 3x3 each.
		model.add(Convolution2D(32, 3, 3, border_mode='full', input_shape=(3, 100, 100)))
		model.add(Activation('relu'))
		model.add(Convolution2D(32, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(64, 3, 3, border_mode='valid'))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		# Note: Keras does automatic shape inference.
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		model.add(Dense(10))
		model.add(Activation('softmax'))
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd)

		model.fit(
		    X, y,
		    show_accuracy=True, validation_split=0.2,
		    batch_size=batch_size, nb_epoch=countof_epoch, verbose=verbose)

		loss, accuracy = model.evaluate(X, y, show_accuracy=True, verbose=verbose)
		print('loss: ', loss)
		print('accuracy: ', accuracy)
		print()
