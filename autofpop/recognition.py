from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
from skimage import io
from skimage import transform
from skimage import color
from skimage.feature import match_template
import os
import glob
from skimage import exposure
from skimage import feature
from skimage import color
import numpy as np
from scipy.spatial import distance
from friendspop import CELL_NAMES
import friendspop
import uuid
from skimage.feature import match_template
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

'''
 candy values:
- 0 blue
- 1 green
- 2 orange
- 3 purple
- 4 yellow'''

DATASET = []
for k, v in CELL_NAMES.items():
    DATASET.append(('Training_Data/'+v, k))

def getColorVector(im, nbin):
    h1, v1 = exposure.histogram(im[:,:,0], nbin)
    h2, v2 = exposure.histogram(im[:,:,1], nbin)
    h3, v3 = exposure.histogram(im[:,:,2], nbin)
    h1 = h1 / (h1.sum() * 1.0)
    h2 = h2 / (h2.sum() * 1.0)
    h3 = h3 / (h3.sum() * 1.0)
    return np.append(h1,[h2,h3])

class ImgRecognizer:
    def __init__(self):
        self.training_data = []
        self.target_values = []
        self.svc = svm.SVC(class_weight='balanced',
                           )
        self.downscale_res = (50, 50)

    def _load(self, path, target_value):
        training_imgs = glob.glob(os.path.join(path, '*.png'))
        for f in training_imgs:
            img = io.imread(f)
            self.training_data.append(self.img2feat(img))
            self.target_values.append(target_value)

    def img2feat(self, img):
        resized = transform.resize(img, self.downscale_res)[:,:,:3]
        colvec  = getColorVector(resized, 5)
        return np.concatenate([
            resized.flatten(),
            colvec,
        ])

    def load(self):
        for (dirname, resp_value) in DATASET:
            if os.path.exists(dirname):
                self._load(dirname, resp_value)

    def train(self):
        if not self.training_data:
            self.load()

        X = np.array(self.training_data)
        y = np.array(self.target_values)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1)
        n_components = 75

        self.pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

        X_train_pca = self.pca.transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        self.clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        self.clf = self.clf.fit(X_train_pca, y_train)
        # print("Best estimator found by grid search:")
        # print(self.clf.best_estimator_)

        # print("Predicting people's names on the test set")
        y_pred = self.clf.predict(X_test_pca)
        # print(classification_report(y_test, y_pred))
        # print(confusion_matrix(y_test, y_pred))
        # self.svc.fit(np_data, np_values)
        # joblib.dump(self.svc, 'svc.dat', compress=9)

    def test(self):
        np_train_data = np.array(self.training_data)
        np_values = np.array(self.target_values)
        data, test_data, train_target, test_target = cross_validation.train_test_split(np_train_data, np_values,
                                                                                       test_size=0.4, random_state=0)
        self.svc.fit(data, train_target)
        print self.svc.score(test_data, test_target)

    def predict_svm(self, im):
        feat = self.img2feat(im)
        result = int(self.svc.predict(feat))
        outdir = os.path.join("data/PredictionLog")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        io.imsave(os.path.join(outdir,  CELL_NAMES[result] + "_" + str(uuid.uuid4()) + ".png"), im)
        return result

    def predict(self, img):
        result = self.predict_(img)
        self.save_image_log(img, result)
        return result

    def predict_(self, img):
        feat = self.img2feat(img)
        afterpca = self.pca.transform(feat.reshape(1, -1))
        result = int(self.clf.predict(afterpca))
        return result

    def save_image_log(self, img, result):
        outdir = os.path.join("data/PredictionLog")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        io.imsave(os.path.join(outdir,  CELL_NAMES[result] + "_" + str(uuid.uuid4()) + ".png"), transform.resize(img, self.downscale_res))
