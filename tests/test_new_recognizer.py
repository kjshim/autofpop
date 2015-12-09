from autofpop.new_recognizer import Image
from autofpop.new_recognizer import ImageReader
from autofpop.new_recognizer import Data
from autofpop.new_recognizer import Recognizer
from autofpop.new_recognizer import RecognizerFlatten
from autofpop.new_recognizer import RecognizerSVM
from autofpop.new_recognizer import RecognizerPCA
from autofpop.new_recognizer import RecognizerLDA
from autofpop.new_recognizer import RecognizerSplit
from numpy.testing import *

import unittest

class ExampleTest(unittest.TestCase):
	def test_basic_usage(self):
		model_filename = 'tmp/basic_usage.pkl'
		recognizer = Recognizer()
		recognizer.load_data([
			'BLACK', 'BLUE', 'BROWN', 'GREEN',
			'PINK', 'WHITE', 'YELLOW',
			'CONE', 'MAPSCROLL', 'STONE',
			'NA',
		])
		recognizer.model = [RecognizerFlatten(), RecognizerSplit(), RecognizerPCA(), RecognizerSVM()]
		recognizer.fit()
		self.assertGreater(recognizer.score(), 0.9)
		recognizer.clear()
		recognizer.dump(model_filename)

		recognizer = Recognizer()
		recognizer.load(model_filename)
		data = Data()
		data.load_data([
			'BLACK', 'BLUE', 'BROWN', 'GREEN',
			'PINK', 'WHITE', 'YELLOW',
			'CONE', 'MAPSCROLL', 'STONE',
			'NA',
		])
		self.assertGreater(recognizer.score(data), 0.97)

	def test_experimental(self):
		recognizer = Recognizer()
		recognizer.load_data([
			'BASE',
			'FLOWER', 'JAIL', 'SNOW',
			'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
			'TRI',
		])
		recognizer.model = [RecognizerFlatten(), RecognizerSplit(), RecognizerPCA(), RecognizerLDA(), RecognizerSVM()]
		recognizer.model[1].n_components = 80
		recognizer.model[2].n_components = 10
		recognizer.fit()
		recognizer.model[1].n_components = 75
		recognizer.model[2].n_components = 30
		self.assertGreater(recognizer.score(), 0.3)


class ImageTest(unittest.TestCase):
	def setUp(self):
		self.subject = Image(
			filename='tests/fixtures/Training_Data/BLACK_BASE/' \
			'BLACK_16c25a59-c736-431f-8fe2-25664299eaea.png')

	def test_with_filename(self):
		self.assertTrue(self.subject.image is not None)

	def test_with_image(self):
		image2 = Image(image=self.subject.image)
		assert_equal(image2.image, self.subject.image)

	def test_feature(self):
		image2 = Image(image=self.subject.image)
		assert_equal(image2.feature(), self.subject.feature())

class ImageReaderTest(unittest.TestCase):
	def setUp(self):
		self.original_base = ImageReader.base
		ImageReader.base = 'tests/fixtures/' + ImageReader.base

		self.subject = ImageReader('BLACK')

	def tearDown(self):
		ImageReader.base = self.original_base

	def test_path_of(self):
		self.assertEqual(
			self.subject.path_of(),
			'tests/fixtures/Training_Data/*BLACK*/*.png')

	def test_filenames(self):
		self.assertEqual(len(self.subject.filenames()), 10)

	def test_read(self):
		for feature in self.subject.read():
			self.assertTrue(feature is not None)

class DataTest(unittest.TestCase):
	def setUp(self):
		self.original_base = ImageReader.base
		ImageReader.base = 'tests/fixtures/' + ImageReader.base

		self.subject = Data()

	def tearDown(self):
		ImageReader.base = self.original_base

	def test_load_data(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.assertEqual(len(self.subject.X), len(self.subject.y))
		self.assertEqual(len(self.subject.X), 15)

class RecognizerTest(unittest.TestCase):
	def setUp(self):
		self.original_base = ImageReader.base
		ImageReader.base = 'tests/fixtures/' + ImageReader.base

		self.subject = Recognizer()

	def tearDown(self):
		ImageReader.base = self.original_base

	def test_fit_predict(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.model = [RecognizerFlatten(), RecognizerSVM()]
		self.subject.fit()
		for X, y in zip(self.subject.data.X, self.subject.data.y):
			self.assertEqual(self.subject.predict(X), y)

	def test_fit_predict_with_PCA_SVM(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.model = [RecognizerFlatten(), RecognizerPCA(), RecognizerSVM()]
		self.subject.fit()
		for X, y in zip(self.subject.data.X, self.subject.data.y):
			self.assertEqual(self.subject.predict(X), y)

	def test_fit_predict_with_Split_PCA_SVM(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.model = [RecognizerFlatten(), RecognizerSplit(), RecognizerPCA(), RecognizerSVM()]
		self.subject.fit()
		self.subject.score()

	def test_fit_predict_with_LDA_SVM(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.model = [RecognizerFlatten(), RecognizerLDA(), RecognizerSVM()]
		self.subject.fit()
		for X, y in zip(self.subject.data.X, self.subject.data.y):
			self.assertEqual(self.subject.predict(X), y)

	def test_dump_load(self):
		model_filename = 'tmp/test_model.pkl'
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.model = [RecognizerFlatten(), RecognizerPCA(), RecognizerSVM()]
		self.subject.fit()
		self.subject.dump(model_filename)

		self.subject = Recognizer()
		self.subject.load(model_filename)
		data = Data()
		data.load_data(['BLACK', 'BLUE'])
		for X, y in zip(data.X, data.y):
			self.assertEqual(self.subject.predict(X), y)

class RecognizerFlattenTest(unittest.TestCase):
	def setUp(self):
		self.original_base = ImageReader.base
		ImageReader.base = 'tests/fixtures/' + ImageReader.base

		self.subject = RecognizerFlatten()

	def tearDown(self):
		ImageReader.base = self.original_base

	def test_fit_predict(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.fit()
		self.assertEqual(
			self.subject.predict(self.subject.data.X).shape[1], 3 * 50 * 50)
		for X, y in zip(self.subject.data.X, self.subject.data.y):
			self.assertEqual(self.subject.predict(X).shape, (3 * 50 * 50, ))

class RecognizerSVMTest(unittest.TestCase):
	def setUp(self):
		self.original_base = ImageReader.base
		ImageReader.base = 'tests/fixtures/' + ImageReader.base

		self.subject = Recognizer()

	def tearDown(self):
		ImageReader.base = self.original_base

	def test_fit_predict(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.model = [RecognizerFlatten(), RecognizerSVM()]
		self.subject.fit()
		for X, y in zip(self.subject.data.X, self.subject.data.y):
			self.assertEqual(self.subject.predict(X), y)

class RecognizerPCATest(unittest.TestCase):
	def setUp(self):
		self.original_base = ImageReader.base
		ImageReader.base = 'tests/fixtures/' + ImageReader.base

		self.subject = Recognizer()

	def tearDown(self):
		ImageReader.base = self.original_base

	def test_fit_predict(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.model = [RecognizerFlatten(), RecognizerPCA()]
		self.subject.fit()
		for X, y in zip(self.subject.data.X, self.subject.data.y):
			self.assertTrue(self.subject.predict(X) is not None)

class RecognizerLDATest(unittest.TestCase):
	def setUp(self):
		self.original_base = ImageReader.base
		ImageReader.base = 'tests/fixtures/' + ImageReader.base

		self.subject = Recognizer()

	def tearDown(self):
		ImageReader.base = self.original_base

	def test_fit_predict(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.model = [RecognizerFlatten(), RecognizerLDA()]
		self.subject.fit()
		for X, y in zip(self.subject.data.X, self.subject.data.y):
			self.assertTrue(self.subject.predict(X) is not None)

class RecognizerSplitTest(unittest.TestCase):
	def setUp(self):
		self.original_base = ImageReader.base
		ImageReader.base = 'tests/fixtures/' + ImageReader.base

		self.subject = RecognizerSplit()

	def tearDown(self):
		ImageReader.base = self.original_base

	def test_fit(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		self.subject.fit()

	def test_transform(self):
		self.subject.load_data(['BLACK', 'BLUE'])
		data = self.subject.transform()
		self.assertEqual(len(data.X), len(data.y))
		self.assertEqual(len(data.X), 12)
		self.assertEqual(
			len(self.subject.test_data.X),
			len(self.subject.test_data.y))
		self.assertEqual(len(self.subject.test_data.X), 3)
