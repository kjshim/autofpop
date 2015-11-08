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
