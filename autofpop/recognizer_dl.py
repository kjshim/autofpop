from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_yaml
from sklearn.externals import joblib

from autofpop.new_recognizer import RecognizerCommon

class RecognizerDL(RecognizerCommon):
	def __init__(self):
		self.batch_size = 128
		self.conv1_filter = 32
		self.conv1_size = 5
		self.conv1_dropout= 0.25
		self.conv2_filter = 64
		self.conv2_size = 5
		self.conv2_dropout= 0.25
		self.dimof_middle = 256
		self.dropout = 0.75
		self.countof_epoch = 500
		self.verbose = 1

	def dump(self, filename):
		open(filename + '.yaml', 'w').write(self.model.to_yaml())
		joblib.dump(self.inverse, filename + '.pkl')
		self.model.save_weights(filename + '.h5', overwrite=True)

	def load(self, filename):
		self.model = model_from_yaml(open(filename + '.yaml').read())
		self.inverse = joblib.load(filename + '.pkl')
		self.model.load_weights(filename + '.h5')

	def fit(self):
		X = self.data.X

		dimof_input = self.data.X[0].shape
		dimof_output = len(set(self.data.y))
		print('dimof_input: ', dimof_input)
		print('dimof_output: ', dimof_output)

		self.inverse, y = np.unique(self.data.y, return_inverse=True)
		y = np_utils.to_categorical(y, dimof_output)

		print('batch_size: ', self.batch_size)
		print('conv1_filter: ', self.conv1_filter)
		print('conv1_size: ', self.conv1_size)
		print('conv1_dropout: ', self.conv1_dropout)
		print('conv2_filter: ', self.conv2_filter)
		print('conv2_size: ', self.conv2_size)
		print('conv2_dropout: ', self.conv2_dropout)
		print('dimof_middle: ', self.dimof_middle)
		print('dropout: ', self.dropout)
		print('countof_epoch: ', self.countof_epoch)
		print('verbose: ', self.verbose)
		print()

		self.model = Sequential()
		self.model.add(Convolution2D(
			self.conv1_filter, self.conv1_size, self.conv1_size,
			border_mode='same', input_shape=dimof_input))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(self.conv1_filter, self.conv1_size, self.conv1_size))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(self.conv1_dropout))

		self.model.add(Convolution2D(self.conv2_filter, self.conv2_size, self.conv2_size))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(self.conv2_filter, self.conv2_size, self.conv2_size))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(self.conv2_dropout))

		self.model.add(Flatten())
		# Note: Keras does automatic shape inference.
		self.model.add(Dense(self.dimof_middle))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(self.dropout))

		self.model.add(Dense(self.dimof_middle))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(self.dropout))

		self.model.add(Dense(dimof_output))
		self.model.add(Activation('softmax'))
		self.model.compile(loss='categorical_crossentropy', optimizer='adadelta')

		self.model.fit(
		    X, y,
		    show_accuracy=True, validation_split=0.2,
		    batch_size=self.batch_size, nb_epoch=self.countof_epoch, verbose=self.verbose)

		loss, accuracy = self.model.evaluate(X, y, show_accuracy=True, verbose=self.verbose)
		print('loss: ', loss)
		print('accuracy: ', accuracy)
		print()

	def predict(self, X):
		return self.inverse[self.model.predict_classes(np.array([X]), verbose=0)]

	def score(self, test=None):
		result = self.inverse[self.model.predict_classes(test.X, verbose=0)] == test.y
		return float(sum(result)) / len(result)
