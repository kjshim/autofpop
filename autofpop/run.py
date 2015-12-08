from __future__ import absolute_import
from __future__ import print_function

from autofpop import andlib
from autofpop import ScreenReader
from autofpop import friendspop
import matplotlib.pyplot as plt

def run():
	andlib.Init()
	while True:
		print("#" * 70)

		sc = andlib.GetScreen()
		sc = ScreenReader.normalizeImage(sc)
		# io.imsave("data/history/hist_" + friendspop.getTimeStr() + ".png", sc)
		mat = ScreenReader.createMatrixFromScreen(sc)
		pprint(mat)
		friendspop.print_board(mat)
		solver = friendspop.SimpleSolver()
		score, [start, end] = solver.solve_board(mat)
		print(score)
		print(start, end)

		score, _, endboard =  solver.check_direction(start, ((end[0] - start[0]), (end[1] - start[1])))

		x1, y1 = ScreenReader.GetCellMidPoint(sc, start[0], start[1])
		x2, y2 = ScreenReader.GetCellMidPoint(sc, end[0], end[1])
		print((x1,y1), (x2,y2))
		plt.imshow(sc)
		plt.plot([x1,x2], [y1,y2], 'k-', lw=2)
		plt.show()

if __name__ == '__main__':
	run()
