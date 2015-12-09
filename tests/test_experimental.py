from autofpop.new_recognizer import Recognizer
from autofpop.new_recognizer import RecognizerFlatten
from autofpop.new_recognizer import RecognizerSVM
from autofpop.new_recognizer import RecognizerPCA
from autofpop.new_recognizer import RecognizerLDA
from autofpop.new_recognizer import RecognizerSplit

import unittest

class ExperimentalTest(unittest.TestCase):
	def test_experimental(self):
		recognizer = Recognizer()
		recognizer.load_data([
			# 'BLACK', 'BLUE', 'BROWN', 'GREEN',
			# 'PINK', 'WHITE', 'YELLOW',
			# 'CONE', 'MAPSCROLL', 'STONE',
			# 'NA',
			'BASE',
			'FLOWER', 'JAIL', 'SNOW',
			'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
			'TRI',
		])
		recognizer.model = [RecognizerFlatten(), RecognizerSplit(), RecognizerPCA(), RecognizerLDA(), RecognizerSVM()]
		result = []
    # for n_pca in [10, 50, 80, 100, 150, 200, 300, 500, 700]:
		for n_pca in [150]:
      # for n_lda in [1, 3, 5, 7, 10, 15, 30, 50, 70, 100, 130, 200, 250, 300]:
			for n_lda in [15]:
				recognizer.model[1].n_components = n_pca
				recognizer.model[2].n_components = n_lda
				n = 10
				score = []
				for i in range(n):
					recognizer.fit()
					score.append(recognizer.score())
				avg = float(sum(score)) / n
				import sys
				print {'n_pca': n_pca, 'n_lda': n_lda, 'score': avg}
				result.append({'n_pca': n_pca, 'n_lda': n_lda, 'score': avg})
		import sys
		print result
		recognizer.model[1].n_components = 75
		recognizer.model[2].n_components = 30
		self.assertGreater(recognizer.score(), 0.5)
		# self.assertGreater(recognizer.score(recognizer.data), 1)
