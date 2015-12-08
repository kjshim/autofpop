from __future__ import absolute_import
from __future__ import print_function

from .new_recognizer import Recognizer
from .new_recognizer import RecognizerSplit
from .recognizer_dl import RecognizerDL

def save_model(name, data):
	recognizer = Recognizer()
	recognizer.load_data(data)
	recognizer.model = [RecognizerSplit(), RecognizerDL()]
	recognizer.model[-1].countof_epoch = 20
	recognizer.fit()
	recognizer.model[-1].dump('model/' + name)

if __name__ == '__main__':
	save_model('color', [
		'BLACK', 'BLUE', 'BROWN', 'GREEN',
		'PINK', 'WHITE', 'YELLOW',
		'CONE', 'MAPSCROLL', 'STONE',
		'NA',
		])
	save_model('type', [
		'BASE',
		'FLOWER', 'JAIL', 'SNOW',
		'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
		'TRI',
		])
