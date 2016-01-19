from __future__ import absolute_import

from autofpop import logic
from autofpop import ScreenReader

import numpy as np

import unittest

import json

def write(object, filename):
	with open(filename, 'w') as f:
		f.write(json.dumps(object))

def read(filename):
	return json.loads(open(filename).read())

class LogicTest(unittest.TestCase):
	def test_matrix(self):
		screen = 'tests/fixtures/screen/origin.png'
		expected = read('tests/fixtures/matrix/matrix.json')
		sc = ScreenReader.read(screen)
		sc = ScreenReader.normalizeImage(sc)
		mat = logic.matrix(sc)
		self.assertEqual(expected, mat)

	def test_solve(self):
		mat = read('tests/fixtures/matrix/matrix.json')
		start, end = logic.solve(mat)
		self.assertEqual((4, 2), start)
		self.assertEqual((4, 3), end)
