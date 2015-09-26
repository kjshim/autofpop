import friendspop
import recognition
import ScreenReader
import matplotlib.pyplot as plt
from skimage import draw

def test1():
    sampledata = [[2, 2, 7, 7, 2, 6, 0, 2, 4],
         [2, 3, 4, 0, 0, 7, 3, 1, 3],
         [1, 5, 7, 5, 7, 5, 0, 5, 3],
         [7, 5, 7, 5, 7, 5, 3, 5, 7],
         [2, 5, 1, 5, 6, 5, 6, 5, 3],
         [2, 5, 6, 5, 1, 5, 3, 5, 4],
         [3, 5, 2, 5, 6, 5, 3, 5, 7],
         [4, 5, 6, 5, 1, 5, 6, 5, 2],
         [4, -1, 6, -1, 6, -1, 2, -1, 2]]
    friendspop.print_board(sampledata)

    solver = friendspop.SimpleSolver()
    print solver.solve_board(sampledata)

def test2():
    sc = ScreenReader.readNormalizedScreen("data/asdf.png")
    mat = ScreenReader.createMatrixFromScreen(sc)
    solver = friendspop.SimpleSolver()
    score, [start, end] = solver.solve_board(mat)
    print(start, end)
    x1, y1 = ScreenReader.GetCellMidPoint(sc, start[0], start[1])
    x2, y2 = ScreenReader.GetCellMidPoint(sc, end[0], end[1])
    print((x1,y1), (x2,y2))
    plt.imshow(sc)
    plt.plot([x1,x2], [y1,y2], 'k-', lw=2)
    plt.show()

test2()