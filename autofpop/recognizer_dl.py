from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
from keras.utils import np_utils # For y values
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

from autofpop.new_recognizer import RecognizerCommon

class RecognizerDL(RecognizerCommon):
	def fit(self):
		X = np.array(self.data.X)

		dimof_input = len(self.data.X[0])
		dimof_output = len(set(self.data.y))
		print('dimof_input: ', dimof_input)
		print('dimof_output: ', dimof_output)

		inverse, y = np.unique(self.data.y, return_inverse=True)
		y = np_utils.to_categorical(y, dimof_output)

		batch_size = 128
		dimof_middle = 1000
		dropout = 0.7
		countof_epoch = 10 #100
		verbose = 1 #0
		print('batch_size: ', batch_size)
		print('dimof_middle: ', dimof_middle)
		print('dropout: ', dropout)
		print('countof_epoch: ', countof_epoch)
		print('verbose: ', verbose)
		print()

		model = Sequential()
		model.add(Dense(dimof_input, dimof_middle, init='uniform', activation='relu'))
		model.add(Dropout(dropout))
		model.add(Dense(dimof_middle, dimof_middle, init='uniform', activation='relu'))
		model.add(Dropout(dropout))
		model.add(Dense(dimof_middle, dimof_middle, init='uniform', activation='relu'))
		model.add(Dropout(dropout))
		model.add(Dense(dimof_middle, dimof_middle, init='uniform', activation='relu'))
		model.add(Dropout(dropout))
		model.add(Dense(dimof_middle, dimof_output, init='uniform', activation='softmax'))
		model.compile(loss='mse', optimizer='sgd')

		model.fit(
		    X, y,
		    show_accuracy=True, validation_split=0.2,
		    batch_size=batch_size, nb_epoch=countof_epoch, verbose=verbose)

		loss, accuracy = model.evaluate(X, y, show_accuracy=True, verbose=verbose)
		print('loss: ', loss)
		print('accuracy: ', accuracy)
		print()
