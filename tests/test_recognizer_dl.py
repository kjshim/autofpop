from autofpop.recognizer_dl import RecognizerDL

import unittest

class ExampleTest(unittest.TestCase):
	def test_basic_usage(self):
		self.subject = RecognizerDL()
		self.subject.load_data([
			'BLACK', 'BLUE', 'BROWN', 'GREEN',
			'PINK', 'WHITE', 'YELLOW',
			'CONE', 'MAPSCROLL', 'STONE',
			'NA',
		])
		self.subject.fit()

		self.fail()