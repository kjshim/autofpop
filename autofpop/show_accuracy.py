from __future__ import absolute_import
from __future__ import print_function

from .recognizer_dl import RecognizerDL

def show_accuracy(name, data):
	recognizer = RecognizerDL()
	recognizer.load_data(data)
	recognizer.load('model/' + name)
	print(name, 'accuracy: ', recognizer.score(recognizer.data))

if __name__ == '__main__':
	show_accuracy('color', [
		'BLACK', 'BLUE', 'BROWN', 'GREEN',
		'PINK', 'WHITE', 'YELLOW',
		# 'CONE', 'MAPSCROLL', 'STONE',
		'NA',
		])
	show_accuracy('type', [
		'BASE',
		'FLOWER', 'JAIL', 'SNOW',
		'STRIPE_1', 'STRIPE_2', 'STRIPE_3',
		'TRI', 'BOMB'
		])
