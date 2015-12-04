from autofpop.new_recognizer import Recognizer
from autofpop.new_recognizer import RecognizerSplit
from autofpop.recognizer_dl import RecognizerDL

import unittest

class ExampleTest(unittest.TestCase):
	def test_basic_usage(self):
		self.subject = Recognizer()
		# self.subject.load_data([
		# 	'BLACK', 'BLUE', 'BROWN', 'GREEN',
		# 	'PINK', 'WHITE', 'YELLOW',
		# 	'CONE', 'MAPSCROLL', 'STONE',
		# 	'NA',
		# ])
		self.subject.load_data([
			'BASE',
			'FLOWER', 'JAIL', 'SNOW',
			'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
			'TRI',
		])
		self.subject.model = [RecognizerSplit(), RecognizerDL()]
		self.subject.fit()
		print(self.subject.score())

		self.fail()
