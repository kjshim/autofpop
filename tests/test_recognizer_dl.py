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
		self.subject.model[-1].batch_size = 128
		self.subject.model[-1].conv1_filter = 2
		self.subject.model[-1].conv1_size = 3
		self.subject.model[-1].conv1_dropout= 0.25
		self.subject.model[-1].conv2_filter = 2
		self.subject.model[-1].conv2_size = 3
		self.subject.model[-1].conv2_dropout= 0.25
		self.subject.model[-1].dimof_middle = 1
		self.subject.model[-1].dropout = 0.5
		self.subject.model[-1].countof_epoch = 1
		self.subject.model[-1].verbose = 1
		self.subject.fit()
		score1 = self.subject.model[-1].score(self.subject.data)

		model_filename = 'tmp/test_model'
		self.subject.model[-1].dump(model_filename)
		self.subject = RecognizerDL()
		self.subject.load(model_filename)
		self.subject.load_data([
			'BASE',
			'FLOWER', 'JAIL', 'SNOW',
			'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
			'TRI',
		])
		score2 = self.subject.score(self.subject.data)

		self.assertEqual(score1, score2)