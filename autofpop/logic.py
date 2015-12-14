from __future__ import absolute_import
from __future__ import print_function

from autofpop import andlib
from autofpop import ScreenReader
from autofpop import friendspop
import matplotlib.pyplot as plt
from pprint import pprint

def matrix(sc):
	mat = ScreenReader.createMatrixFromScreen(sc)
	pprint(mat)
	friendspop.print_board(mat)
	return mat

def solve(mat):
	solver = friendspop.SimpleSolver()
	score, [start, end] = solver.solve_board(mat)
	print(score)
	print(start, end)

	score, _, endboard =  solver.check_direction(start, ((end[0] - start[0]), (end[1] - start[1])))
	return start, end

def show(sc, start, end):
	x1, y1 = ScreenReader.GetCellMidPoint(sc, start[0], start[1])
	x2, y2 = ScreenReader.GetCellMidPoint(sc, end[0], end[1])
	print((x1,y1), (x2,y2))
	plt.imshow(sc)
	plt.plot([x1,x2], [y1,y2], 'k-', lw=2)
	plt.show()

def run():
	andlib.Init()
	while True:
		print("#" * 70)
		sc = andlib.GetScreen()
		sc = ScreenReader.normalizeImage(sc)
		mat = matrix(sc)
		start, end = solve(mat)
		show(sc, start, end)
