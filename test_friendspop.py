import friendspop

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

test1()