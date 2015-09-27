import friendspop
import recognition
import ScreenReader
import matplotlib.pyplot as plt
from skimage import draw
from skimage import io
import andlib
import datetime
from pprint import pprint
import glob, os
from copy import deepcopy

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
        print "##################################################################################################"
        try:
            filelist = glob.glob("data/PredictionLog/*.png")
            for f in filelist:
                os.remove(f)
        except:
            pass

        sc = andlib.GetScreen()
        sc = ScreenReader.normalizeImage(sc)
        # io.imsave("data/history/hist_" + friendspop.getTimeStr() + ".png", sc)
        mat = ScreenReader.createMatrixFromScreen(sc)
        pprint(mat)
        friendspop.print_board(mat)
        solver = friendspop.SimpleSolver()
        score, [start, end] = solver.solve_board(mat)
        print score
        print(start, end)

        friendspop.DEBUG_EXPLOSIONS = True
        score, _, endboard =  solver.check_direction(start, ((end[0] - start[0]), (end[1] - start[1])))
        friendspop.DEBUG_EXPLOSIONS = False

        x1, y1 = ScreenReader.GetCellMidPoint(sc, start[0], start[1])
        x2, y2 = ScreenReader.GetCellMidPoint(sc, end[0], end[1])
        print((x1,y1), (x2,y2))
        plt.imshow(sc)
        plt.plot([x1,x2], [y1,y2], 'k-', lw=2)
        plt.show()

def testDebugLogic():
    mat = [[4, 3, 7, 4, 4, 1, 1, 7, 3],
             [6, 1, 0, 3, 0, 3, 6, 1, 4],
             [1, 4, 7, 0, 4, 3, 0, 1, 1],
             [4, 1, 7, 3, 6, 1, 4, 3, 4],
             [1, 1, 2, 18, 3, 2, 6, 1, 3],
             [6, 7, 7, 6, 4, 1, 0, 7, 3],
             [0, 3, 0, 4, 24, 6, 0, 0, 6],
             [3, 4, 0, 3, 7, 1, 7, 0, 3],
             [-1, -1, 6, -1, 1, -1, 5, -1, -1]]
    # friendspop.DEBUG_EXPLOSIONS = True
    solver = friendspop.SimpleSolver()
    solver.game_board = mat
    max_score, chosen_move = solver.solve_board(deepcopy(mat))

    friendspop.print_board(mat, chosen_move)
    print max_score, chosen_move
    solver.game_board = mat

    friendspop.DEBUG_EXPLOSIONS = True

    chosen_move = [( 7, 4), (6,5)]
    score, _, endboard =  solver.check_direction(chosen_move[0], ((chosen_move[1][0] - chosen_move[0][0]), (chosen_move[1][1] - chosen_move[0][1])))
    print score
    friendspop.print_board(endboard)

def testDebugLogic2():
    mat = [[4, 3, 7, 4, 4, 1, 1, 7, 3],
             [6, 1, 0, 3, 0, 3, 6, 1, 4],
             [1, 4, 7, 0, 4, 3, 0, 1, 1],
             [4, 1, 7, 3, 6, 1, 4, 3, 4],
             [1, 1, 2, 18, 3, 2, 6, 1, 3],
             [6, 7, 7, 6, 4, 1, 0, 7, 3],
             [0, 3, 0, 4, 24, 6, 0, 0, 6],
             [3, 4, 0, 3, 7, 1, 7, 0, 3],
             [-1, -1, 6, -1, 1, -1, 5, -1, -1]]
    # friendspop.DEBUG_EXPLOSIONS = True
    solver = friendspop.SimpleSolver()
    solver.game_board = mat
    friendspop.DEBUG_EXPLOSIONS = True
    chosen_move = [( 7, 4), (6,5)]
    score, _, endboard =  solver.check_direction(chosen_move[0], ((chosen_move[1][0] - chosen_move[0][0]), (chosen_move[1][1] - chosen_move[0][1])))
    print score
    friendspop.print_board(endboard)
# testUsingPhone()
testDebugLogic2()