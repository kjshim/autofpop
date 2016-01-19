from autofpop import recognition
from autofpop.friendspop import CELL_NAMES
from skimage import io
import os
import glob

def test_ImgRecognizer_example():
	root = 'Training_Data'
	klass = 'BLACK_BASE'
	def value_of(name):
	    return next(
	        (key for key, value in CELL_NAMES.items() if value == klass),
	        None)
	model = recognition.ImgRecognizer()
	model.load()
	model.train()

	path = glob.glob(os.path.join(root, klass, '*.png'))[0]
	img = io.imread(path)

	result = model.predict_(img)
	expected = value_of(klass)
	print result, expected
	# assert result == expected

from skimage import transform
from skimage import io
from skimage import exposure
from sklearn.decomposition import RandomizedPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import numpy as np
import os
import glob

class Recognizer(object):
	def __init__(self, klasses):
		self.base = 'Training_Data'
		self.downscale_res = (50, 50)
		self.n_components = 75
		self.simplify = self.pca_
		self.param_grid = {
			'C': [1e3, 5e3, 1e4, 5e4, 1e5],
			'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
		}

		self.X = []
		self.y = []

		self.klasses = klasses

	def path_of(self, klass):
		return os.path.join(self.base, '*' + klass + '*', '*.png')

	def img(self, fn):
		return io.imread(fn)


	def feature(self, img):
		def getColorVector(im, nbin):
			h1, v1 = exposure.histogram(im[:,:,0], nbin)
			h2, v2 = exposure.histogram(im[:,:,1], nbin)
			h3, v3 = exposure.histogram(im[:,:,2], nbin)
			h1 = h1 / (h1.sum() * 1.0)
			h2 = h2 / (h2.sum() * 1.0)
			h3 = h3 / (h3.sum() * 1.0)
			return np.append(h1,[h2,h3])
		resized = transform.resize(img, self.downscale_res)[:,:,:3]
		colvec  = getColorVector(resized, 5)
		return np.concatenate([
			resized.flatten(),
			colvec,
		])

	def load(self):
		for klass in self.klasses:
			for fn in glob.glob(self.path_of(klass)):
				self.X.append(self.feature(self.img(fn)))
				self.y.append(klass)

	def pca_(self, X, y, n_components):
		return RandomizedPCA(n_components=n_components, whiten=True).fit(X)

	def lda_(self, X, y, n_components):
		return LinearDiscriminantAnalysis(n_components=n_components).fit(X, y)

	def train_(self, X, y):
		simple = self.simplify(X, y, self.n_components)
		X_simple = simple.transform(X)
		clf = GridSearchCV(
			SVC(kernel='rbf', class_weight='balanced'),
			self.param_grid)
		clf = clf.fit(X_simple, y)
		return simple, clf

	def train(self):
		X = np.array(self.X)
		y = np.array(self.y)
		X_train, self.X_test, y_train, self.y_test = \
			train_test_split(
				X, y, test_size=0.2, random_state=0)
		self.simple, self.clf = self.train_(X_train, y_train)

	def dump(self):
		X_test_simple = self.simple.transform(self.X_test)
		y_pred = self.clf.predict(X_test_simple)
		# print self.clf.best_estimator_
		# print classification_report(self.y_test, y_pred)
		# print confusion_matrix(self.y_test, y_pred)
		print self.clf.score(X_test_simple, self.y_test)

	def predict(self, img):
		X = self.feature(img)
		X_simple = self.simple.transform(X)
		return self.clf.predict(X_simple)

import random

def get_one(model, klass):
	return random.choice(
		glob.glob(model.path_of(klass))
	)

def xtest_Recognizer():
	klasses = [
		'BLACK', 'BLUE', 'BROWN', 'GREEN',
		'PINK', 'WHITE', 'YELLOW',
		'CONE', 'MAPSCROLL', 'STONE',
		'NA',
	]
	model = Recognizer(klasses)
	model.load()
	model.train()

	for klass in klasses:
		img = model.img(get_one(model, klass))
		actual = model.predict(img)
		print klass, actual
		# assert len(actual) == 1
		# assert klass == actual[0]

def xtest_Recognizer2():
	klasses = [
		'BASE',
		'FLOWER', 'JAIL', 'SNOW',
		'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
		'TRI',
	]
	model = Recognizer(klasses)
	model.load()
	model.train()

	for klass in klasses:
		img = model.img(get_one(model, klass))
		actual = model.predict(img)
		print klass, actual
		# assert len(actual) == 1
		# assert klass == actual[0]

def xtest_Recognizer3():
	type1 = Recognizer([
		'BLACK', 'BLUE', 'BROWN', 'GREEN',
		'PINK', 'WHITE', 'YELLOW',
		'CONE', 'MAPSCROLL', 'STONE',
		'NA',
	])
	type2 = Recognizer([
		'BASE',
		'FLOWER', 'JAIL', 'SNOW',
		'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
		'TRI',
	])
	type1.load()
	type1.train()
	type2.load()
	type2.train()

	for i in range(30):
		path = get_one(type1, '')
		img = type1.img(path)
		t1 = type1.predict(img)
		t2 = type2.predict(img)
		print path, t1, t2
		# assert path.startswith('Training_Data/' + t1[0] + '_' + t2[0] + '/')
	# assert False

def xtest_Recognizer4():
	'''
	type1 = Recognizer([
		'BLACK', 'BLUE', 'BROWN', 'GREEN',
		'PINK', 'WHITE', 'YELLOW',
		'CONE', 'MAPSCROLL', 'STONE',
		'NA',
	])
	type1.load()
	type1.train()
	type1.dump()
	'''

	klasses2 = [
		'BASE',
		'FLOWER', 'JAIL', 'SNOW',
		'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
		'TRI',
	]

	type2 = Recognizer(klasses2)
	type2.n_components = 75
	type2.simplify = type2.pca_
	# type2.load(); type2.train(); type2.dump()

	type2 = Recognizer(klasses2)
	type2.n_components = 75 * 2
	type2.simplify = type2.pca_
	# type2.load(); type2.train(); type2.dump()

	type2 = Recognizer(klasses2)
	type2.n_components = 75
	type2.simplify = type2.lda_
	# type2.load(); type2.train(); type2.dump()

	type2 = Recognizer(klasses2)
	type2.n_components = 75 * 2
	type2.simplify = type2.lda_
	# type2.load(); type2.train(); type2.dump()

	type2 = Recognizer(klasses2)
	type2.n_components = None
	type2.simplify = type2.lda_
	# type2.load(); type2.train(); type2.dump()

	type2 = Recognizer(klasses2)
	type2.n_components = 75
	type2.simplify = type2.lda_
	type2.load(); type2.train(); type2.dump()

	# assert False
