__author__ = 'Kyujin'
import math
import matplotlib.pyplot as plt
import andlib
from scipy import misc
from skimage import io
from skimage.color import rgb2gray
from skimage import transform

## train model
import recognition

CUTMODE="xiaomi"
# CUTMODE="lucy"

SCREEN_NORMALIZE_SIZE = (960, 540)
CELL_NORMALIZE_SIZE = (50,50)

N_COL = 9
N_ROW = 9
def GetCellImage(screenNorm, row, col):
    height, width = screenNorm.shape[:2]
    cellHeight = height / N_ROW
    cellWidth  = width  / N_COL
    y1 = int(round(cellHeight * row + (col%2) * cellHeight / 2.0))
    y2 = int(round(y1 + cellHeight))
    x1 = int(round(cellWidth * col))
    x2 = int(round(x1 + cellWidth))
    return screenNorm[y1:y2, x1:x2, :]

def GetCellMidPoint(screenNorm, row, col):
    height, width = screenNorm.shape[:2]
    cellHeight = height / N_ROW
    cellWidth  = width  / N_COL
    y1 = int(round(cellHeight * row + (col%2) * cellHeight / 2.0))
    y2 = int(round(y1 + cellHeight))
    x1 = int(round(cellWidth * col))
    x2 = int(round(x1 + cellWidth))
    return ((x1+x2)/2., (y1+y2)/2.)

def readNormalizedScreen(fname):
    screen=io.imread(fname)[:,:,:3] # remove alpha
    return normalizeImage(screen)

def normalizeImage(screen):
    if(CUTMODE == "xiaomi"):
        scNormalized = transform.resize(screen, SCREEN_NORMALIZE_SIZE)[204:805,4:539,:]
    if(CUTMODE == "lucy"):
        scNormalized = transform.resize(screen, SCREEN_NORMALIZE_SIZE)[172:772,4:539,:]
    return scNormalized

def read(fname):
    return io.imread(fname)[:,:,:3] # remove alpha

import datetime

def getTimeStr():
    dt=datetime.datetime.now()
    return dt.strftime('%Y%m%d%H%M%S')

def write(sc, fname=None):
    if not fname:
        fname = 'data/screen/' + getTimeStr() + '.png'
    io.imsave(fname, sc)

from .friendspop import CELL_NAMES
from .recognizer_dl import RecognizerDL
from .new_recognizer import Image
import numpy as np

def value_of(name):
    return next(
        (key for key, value in CELL_NAMES.items() if value == name),
        None)

recognizer_color = RecognizerDL()
recognizer_color.load('model/color')
recognizer_type = RecognizerDL()
recognizer_type.load('model/type')

def predict(image):
    image = np.array(Image(image=image).feature())
    # TODO: Save image
    klass = '_'.join(
        recognizer_color.predict(image).tolist() + 
        recognizer_type.predict(image).tolist())
    return value_of(klass)

def createMatrixFromScreen(scNormalized):
    matrix = [[-1 for x in range(9)] for x in range(9)]
    for j in range(9):
        for i in range(9 - (j%2)):
            matrix[i][j] = predict(GetCellImage(scNormalized, i,j))

    return matrix
