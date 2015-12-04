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
		X = self.data.X

		dimof_input = self.data.X[0].shape
		dimof_output = len(set(self.data.y))
		print('dimof_input: ', dimof_input)
		print('dimof_output: ', dimof_output)

		self.inverse, y = np.unique(self.data.y, return_inverse=True)
		y = np_utils.to_categorical(y, dimof_output)

		batch_size = 64
		dimof_middle = 64
		dropout = 0.5
		countof_epoch = 30
		verbose = 1
		print('batch_size: ', batch_size)
		print('dimof_middle: ', dimof_middle)
		print('dropout: ', dropout)
		print('countof_epoch: ', countof_epoch)
		print('verbose: ', verbose)
		print()

		self.model = Sequential()
		self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=dimof_input))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(32, 3, 3))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		# self.model.add(Convolution2D(16, 3, 3))
		# self.model.add(Activation('relu'))
		# self.model.add(Convolution2D(16, 3, 3))
		# self.model.add(Activation('relu'))
		# self.model.add(MaxPooling2D(pool_size=(2, 2)))
		# self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		# Note: Keras does automatic shape inference.
		self.model.add(Dense(dimof_middle))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(dropout))

		self.model.add(Dense(16))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(dropout))

		self.model.add(Dense(dimof_output))
		self.model.add(Activation('softmax'))
		self.model.compile(loss='categorical_crossentropy', optimizer='adadelta')

		self.model.fit(
		    X, y,
		    show_accuracy=True, validation_split=0.2,
		    batch_size=batch_size, nb_epoch=countof_epoch, verbose=verbose)

		loss, accuracy = self.model.evaluate(X, y, show_accuracy=True, verbose=verbose)
		print('loss: ', loss)
		print('accuracy: ', accuracy)
		print()

	def predict(self, X):
		return self.inverse[self.model.predict_classes(np.array([X]), verbose=0)]

	def score(self, test=None):
		result = self.inverse[self.model.predict_classes(test.X, verbose=0)] == test.y
		return float(sum(result)) / len(result)
