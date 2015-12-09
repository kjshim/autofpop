from __future__ import absolute_import
from __future__ import print_function

from autofpop import recognition
from autofpop.friendspop import CELL_NAMES
from skimage import io
import os
import glob

from autofpop.recognizer_dl import RecognizerDL
from autofpop.new_recognizer import ImageReader
import numpy as np

import unittest

class FitRecognizersTest(unittest.TestCase):
	def value_of(self, name):
	    return next(
	        (key for key, value in CELL_NAMES.items() if value == name),
	        None)

	def xtest_ImgRecognizer_example(self):
		root = 'Training_Data'
		klass = 'BLACK_BASE'
		def value_of(name):
		    return next(
		        (key for key, value in CELL_NAMES.items() if value == name),
		        None)
		model = recognition.ImgRecognizer()
		model.load()
		model.train()

		path = glob.glob(os.path.join(root, klass, '*.png'))[0]
		img = io.imread(path)

		result = model.predict_(img)
		expected = value_of(klass)
		self.assertEqual(result, expected)

	def predict(self, img):
		recognizer_color = RecognizerDL()
		recognizer_color.load('model/color')
		recognizer_type = RecognizerDL()
		recognizer_type.load('model/type')

		img = np.array(img)

		klass = '_'.join(
			recognizer_color.predict(img).tolist() +
			recognizer_type.predict(img).tolist())
		return self.value_of(klass)

	def test_DL_example(self):
		img = ImageReader('BLACK_BASE').read()[0]
		result = self.predict(img)
		print(result)
		result = self.predict(img)
		print(result)
