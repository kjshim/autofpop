from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
from skimage import io
from skimage import transform
from skimage import color
from skimage.feature import match_template
import os
from skimage import exposure
import numpy as np
from scipy.spatial import distance
from friendspop import CELL_NAMES
import friendspop
import uuid
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
        self.svc = svm.SVC(gamma=0.001, kernel='linear', C=100)
        self.downscale_res = (50, 50)

    def _load(self, path, target_value):
        training_imgs = os.listdir(path)
        for f in training_imgs:
            img = io.imread(path+'/'+f)
            img = transform.resize(img, self.downscale_res)
            self.training_data.append(img)
            self.target_values.append(target_value)

    def load(self):
        for (dirname, resp_value) in DATASET:
            if os.path.exists(dirname):
                self._load(dirname, resp_value)

    def train(self):
        pass

    def predict(self, img):
        resized_img = transform.resize(img, self.downscale_res)
        maxvv = 1000
        maxi = -1
        color_img = getColorVector(resized_img, 5)
        for i, td in enumerate(self.training_data):
            cc = getColorVector(td, 5)
            vv = distance.euclidean(color_img, cc)
            if vv < maxvv:
                maxvv = vv
                maxi = i

        result = self.target_values[maxi]
        outdir = os.path.join("data/PredictionLog")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        io.imsave(os.path.join(outdir,  CELL_NAMES[result] + "_" + str(uuid.uuid4()) + ".png"), resized_img)

        return result

