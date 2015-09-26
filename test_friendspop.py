import friendspop
import recognition
import ScreenReader
import matplotlib.pyplot as plt
from skimage import draw
from skimage import io
import andlib
import datetime
from pprint import pprint

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
    print score
    print(start, end)
    x1, y1 = ScreenReader.GetCellMidPoint(sc, start[0], start[1])
    x2, y2 = ScreenReader.GetCellMidPoint(sc, end[0], end[1])
    print((x1,y1), (x2,y2))
    plt.imshow(sc)
    plt.plot([x1,x2], [y1,y2], 'k-', lw=2)
    plt.show()

def testUsingPhone():
    andlib.Init()
    while True:
        sc = andlib.GetScreen()
        sc = ScreenReader.normalizeImage(sc)
        io.imsave("data/history/hist_" + friendspop.getTimeStr() + ".png", sc)
        mat = ScreenReader.createMatrixFromScreen(sc)
        pprint(mat)
        friendspop.print_board(mat)
        solver = friendspop.SimpleSolver()
        score, [start, end] = solver.solve_board(mat)
        print score
        print(start, end)
        x1, y1 = ScreenReader.GetCellMidPoint(sc, start[0], start[1])
        x2, y2 = ScreenReader.GetCellMidPoint(sc, end[0], end[1])
        print((x1,y1), (x2,y2))
        plt.imshow(sc)
        plt.plot([x1,x2], [y1,y2], 'k-', lw=2)
        plt.show()

def testDebugLogic():
    mat =   [[5, 5, -1, 5, 5, 5, 5, 5, 5],
             [-1, 7, -1, 3, -1, 4, -1, 3, -1],
             [-1, 6, 4, 2, 15, 4, 4, 7, -1],
             [-1, 3, 7, 17, 6, 3, 6, 3, -1],
             [-1, 1, 10, 6, 18, 1, 1, 1, -1],
             [-1, 4, 3, 1, 15, 2, 6, 3, -1],
             [-1, 3, 4, 6, 1, 6, 6, 1, -1],
             [-1, -1, -1, -1, -1, -1, -1, -1, -1],
             [12, -1, 5, -1, 5, -1, 5, -1, 5]]

    solver = friendspop.SimpleSolver()
    solver.game_board = mat
    solver.check_direction((3, 7), (1, 0))
testUsingPhone()
# testDebugLogic()