from autofpop import recognition
from autofpop.friendspop import CELL_NAMES
from skimage import io
import os
import glob

def test_ImgRecognizer_example():
	def value_of(name):
	    return next(
	        (key for key, value in CELL_NAMES.items() if value == 'BLACK'),
	        None)
	model = recognition.ImgRecognizer()
	model.load()
	model.train()

	root = 'Training_Data'
	klass = 'BLACK'
	path = glob.glob(os.path.join(root, klass, '*.png'))[0]
	img = io.imread(path)

	result = model.predict_(img)
	expected = value_of(klass)
	assert result == expected

from skimage import transform
from skimage import io
from skimage import exposure
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import os
import glob

class Recognizer(object):
	def __init__(self, klasses):
		self.base = 'Training_Data'
		self.downscale_res = (50, 50)
		self.n_components = 75
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

	def pca_(self, X, n_components):
		return RandomizedPCA(n_components=n_components, whiten=True).fit(X)

	def train_(self, X, y):
		pca = self.pca_(X, self.n_components)
		X_pca = pca.transform(X)
		clf = GridSearchCV(
			SVC(kernel='rbf', class_weight='auto'),
			self.param_grid)
		clf = clf.fit(X_pca, y)
		return pca, clf

	def train(self):
		X = np.array(self.X)
		y = np.array(self.y)
		self.pca, self.clf = self.train_(
			np.array(self.X),
			np.array(self.y)
		)

	def predict(self, img):
		X = self.feature(img)
		X_pca = self.pca.transform(X)
		return self.clf.predict(X_pca)

import random

def get_one(model, klass):
	return random.choice(
		glob.glob(model.path_of(klass))
	)

def test_Recognizer():
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
		print klass, model.predict(img)

def test_Recognizer2():
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
		print klass, model.predict(img)

def test_Recognizer3():
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
		print path, type1.predict(img), type2.predict(img)
