from skimage import io
from skimage import exposure
from skimage import transform
import numpy as np
import os
import glob

from keras.preprocessing import image

class Image(object):
	downscale_res = (50, 50)

	def __init__(self, filename=None, image=None):
		self.image = image
		if filename:
			self.image = self.read(filename)

	@staticmethod
	def read(filename):
		return io.imread(filename)

	def feature(self):
		resized = transform.resize(self.image, self.downscale_res)[:,:,:3]
		changed = np.zeros(
			(
				resized.shape[2],
				resized.shape[0],
				resized.shape[1]
			), dtype="float32")
		for channel in xrange(changed.shape[0]):
			changed[channel, :, :] = resized[:, :, channel]
		return changed

class ImageReader(object):
	base = 'Training_Data'

	def __init__(self, klass):
		self.klass = klass

	def path_of(self):
		return os.path.join(self.base, '*' + self.klass + '*', '*.png')

	def filenames(self):
		return glob.glob(self.path_of())

	def read(self):
		return [Image(filename=fn).feature() for fn in self.filenames()]

class Data(object):
	def load_data(self, klasses):
		X = []
		self.y = []
		for klass in klasses:
			for feature in ImageReader(klass).read():
				X.append(feature)
				self.y.append(klass)
		self.X = np.zeros((len(X), ) + X[0].shape, dtype="float32")
		for i, x in enumerate(X):
			self.X[i, :, :, :] = x

from sklearn.externals import joblib

class RecognizerCommon(object):
	def load_data(self, klasses):
		self.data = Data()
		self.data.load_data(klasses)

	def clear(self):
		try:
			del(self.data)
		except AttributeError:
			pass
		try:
			del(self.test_data)
		except AttributeError:
			pass

	def dump(self, filename):
		joblib.dump(self.model, filename)

	def load(self, filename):
		self.model = joblib.load(filename)

	def transform(self):
		return self.transform_(self.data)

	def transform_(self, data):
		data_ = Data()
		data_.X = self.predict(data.X)
		data_.y = data.y
		return data_

	def score(self, test):
		return self.transform_(test)

	def correct_dim(self, X):
		if len(X.shape) < 2:
			return X.reshape((-1, ) + X.shape)
		return X

class Recognizer(RecognizerCommon):
	def clear(self):
		for recognizer in self.model:
			recognizer.clear()

	def fit(self):
		data = Data()
		data.X = self.data.X
		data.y = self.data.y

		for recognizer in self.model[:-1]:
			recognizer.data = data
			recognizer.fit()
			data = recognizer.transform()

		self.model[-1].data = data
		self.model[-1].fit()

	def predict(self, X):
		for recognizer in self.model:
			X = recognizer.predict(X)
		return X

	def score(self, test=None):
		if not test:
			test = Data()
			test.X = np.array([])
			test.y = np.array([])
		for recognizer in self.model:
			test = recognizer.score(test)
		return test

class RecognizerFlatten(RecognizerCommon):
	def fit(self):
		pass

	def predict(self, X):
		return X.reshape(X.shape[:-3] + (-1, ))

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

class RecognizerSVM(RecognizerCommon):
	param_grid = {
		'C': [1e3, 5e3, 1e4, 5e4, 1e5],
		'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
	}

	def fit(self):
		self.model = GridSearchCV(
			SVC(kernel='rbf', class_weight='balanced'),
			self.param_grid).fit(self.data.X, self.data.y)

	def predict(self, X):
		return self.model.predict(self.correct_dim(X))

	def score(self, test):
		return self.model.score(test.X, test.y)

from sklearn.cross_validation import train_test_split

class RecognizerSplit(RecognizerCommon):
	test_size = 0.2
	def fit(self):
		pass

	def transform(self):
		data = Data()
		self.test_data = Data()
		data.X, self.test_data.X, data.y, self.test_data.y = \
			train_test_split(self.data.X, self.data.y, test_size=self.test_size)
		return data

	def predict(self, X):
		return X

	def score(self, test):
		try:
			return self.test_data
		except:
			return test

from sklearn.decomposition import RandomizedPCA

class RecognizerPCA(RecognizerCommon):
	n_components = 75
	def fit(self):
		self.model = RandomizedPCA(
			n_components=self.n_components, whiten=True).fit(self.data.X)

	def predict(self, X):
		return self.model.transform(self.correct_dim(X))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class RecognizerLDA(RecognizerCommon):
	n_components = 30
	def fit(self):
		self.model = LinearDiscriminantAnalysis(
			n_components=self.n_components).fit(self.data.X, self.data.y)

	def predict(self, X):
		return self.model.transform(self.correct_dim(X))
