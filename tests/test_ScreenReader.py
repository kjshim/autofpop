from __future__ import absolute_import

from autofpop import logic
from autofpop import andlib
from autofpop import ScreenReader
from autofpop.new_recognizer import Image, ImageReader

import numpy as np

import unittest

class ScreenReaderTest(unittest.TestCase):
	def test_value_of(self):
		self.assertEqual(0, ScreenReader.value_of('BLACK_BASE'))
		self.assertEqual(-1, ScreenReader.value_of('NA_BASE'))
		self.assertEqual(4, ScreenReader.value_of('PINK_BASE'))

	def test_predict(self):
		image = Image(filename=ImageReader('BLACK_BASE').filenames()[0]).image
		ScreenReader.predict(image)
		ScreenReader.predict(image)
		ScreenReader.predict(image)

	def test_write_and_read(self):
		fixture = 'tests/fixtures/screen/origin.png'
		output = 'tmp/screen.png'
		sc = ScreenReader.read(fixture)
		ScreenReader.write(sc, output)
		sc2 = ScreenReader.read(output)
		self.assertTrue(np.array_equal(sc, sc2))
