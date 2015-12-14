from copy import deepcopy

DEBUG_EXPLOSIONS = False
CELL_NAMES = {
    -1: "NA_BASE",
    0:"BLACK_BASE",
    1:"BLUE_BASE",
    2:"BROWN_BASE",
    3:"GREEN_BASE",
    4:"PINK_BASE",
    5:"STONE_BASE",
    6:"WHITE_BASE",
    7:"YELLOW_BASE",
    8:"BLACK_JAIL",
    9:"BLUE_JAIL",
    10:"BROWN_JAIL",
    11:"GREEN_JAIL",
    12:"PINK_JAIL",
    14:"WHITE_JAIL",
    15:"YELLOW_JAIL",
    16:"MAPSCROLL_BASE",
    
    17:"BLACK_STRIPE_1",
    18:"BLUE_STRIPE_1",
    19:"BROWN_STRIPE_1",
    20:"GREEN_STRIPE_1",
    21:"PINK_STRIPE_1",
    23:"WHITE_STRIPE_1",
    24:"YELLOW_STRIPE_1",

    25:"BLACK_STRIPE_2",
    26:"BLUE_STRIPE_2",
    27:"BROWN_STRIPE_2",
    28:"GREEN_STRIPE_2",
    29:"PINK_STRIPE_2",
    31:"WHITE_STRIPE_2",
    32:"YELLOW_STRIPE_2",

    33:"BLACK_STRIPE_3",
    34:"BLUE_STRIPE_3",
    35:"BROWN_STRIPE_3",
    36:"GREEN_STRIPE_3",
    37:"PINK_STRIPE_3",
    39:"WHITE_STRIPE_3",
    40:"YELLOW_STRIPE_3",

    43:"BLACK_FLOWER",
    44:"BLUE_FLOWER",
    45:"BROWN_FLOWER",
    46:"GREEN_FLOWER",
    47:"PINK_FLOWER",
    49:"WHITE_FLOWER",
    50:"YELLOW_FLOWER",

    53:"BLACK_SNOW",
    54:"BLUE_SNOW",
    55:"BROWN_SNOW",
    56:"GREEN_SNOW",
    57:"PINK_SNOW",
    59:"WHITE_SNOW",
    60:"YELLOW_SNOW",

    63:"BLACK_TRI",
    64:"BLUE_TRI",
    65:"BROWN_TRI",
    66:"GREEN_TRI",
    67:"PINK_TRI",
    69:"WHITE_TRI",
    70:"YELLOW_TRI",

    71:"BLACK_BOMB",
    72:"BLUE_BOMB",
    73:"BROWN_BOMB",
    74:"GREEN_BOMB",
    75:"PINK_BOMB",
    76:"WHITE_BOMB",
    77:"YELLOW_BOMB",

    100:"CONE_BASE",
}

CELL_NAME_TO_VALUE = dict([(v,k) for k,v in CELL_NAMES.items()])
POSTFIX = ["_BASE", "_JAIL", "_STRIPE_1", "_STRIPE_2", "_STRIPE_3", "_FLOWER", "_SNOW", "_TRI", "_BOMB"]
COLORS  = ["BLACK", "BLUE", "BROWN", "PINK", "WHITE", "YELLOW", "GREEN"]
MATCH_LIST_NAME = [
    [ color + post for post in POSTFIX]
        for color in COLORS ]

MATCH_LIST = [ [CELL_NAME_TO_VALUE[v] for v in ml] for ml in MATCH_LIST_NAME]

def print_board(board, highlights=[]):
    for i, line in enumerate(board):
        for j, elem in enumerate(line):
            if(j%2 == 0):
                if (i,j) in highlights:
                    print "*%6s*"%CELL_NAMES[elem] + '\t\t',
                else:
                    print "%8s"%CELL_NAMES[elem] + '\t\t',

            else:
                print '\t\t\t',
        print
        for j, elem in enumerate(line):
            if(j%2 == 1):
                if (i,j) in highlights:
                    print "*%6s*"%CELL_NAMES[elem] + '\t\t',
                else:
                    print "%8s"%CELL_NAMES[elem] + '\t\t',
            else:
                print '\t\t\t',
        print

def getU(pos):
    return (pos[0] - 1, pos[1])

def getD(pos):
    return (pos[0] + 1, pos[1])

def getUL(pos):
    if(pos[1]%2 == 0):
        return (pos[0] -1, pos[1] - 1)
    else:
        return (pos[0], pos[1] - 1)

def getDL(pos):
    if(pos[1]%2 == 0):
        return (pos[0], pos[1] - 1)
    else:
        return (pos[0] + 1, pos[1] - 1)

def getUR(pos):
    if(pos[1]%2 == 0):
        return (pos[0] -1, pos[1] + 1)
    else:
        return (pos[0], pos[1] + 1)

def getDR(pos):
    if(pos[1]%2 == 0):
        return (pos[0], pos[1] + 1)
    else:
        return (pos[0] + 1, pos[1] + 1)

class SimpleSolver:
    def __init__(self):
        self.board_size = 9
        self.match_list = MATCH_LIST

        self.simple_candies = [CELL_NAME_TO_VALUE[v] for v in ["BLACK_BASE","BLUE_BASE","BROWN_BASE","GREEN_BASE","PINK_BASE","WHITE_BASE","YELLOW_BASE"]]
        self.striped_candies_h = [CELL_NAME_TO_VALUE[v] for v in CELL_NAMES.values() if "_STRIPE_1" in v ]
        self.striped_candies_v1 = [CELL_NAME_TO_VALUE[v] for v in CELL_NAMES.values() if "_STRIPE_2" in v ]
        self.striped_candies_v2 = [CELL_NAME_TO_VALUE[v] for v in CELL_NAMES.values() if "_STRIPE_3" in v ]
        self.flower_candies = [CELL_NAME_TO_VALUE[v] for v in CELL_NAMES.values() if "_FLOWER" in v ]
        self.snow_candies = [CELL_NAME_TO_VALUE[v] for v in CELL_NAMES.values() if "_SNOW" in v ]
        self.tri_candies = [CELL_NAME_TO_VALUE[v] for v in CELL_NAMES.values() if "_TRI" in v ]
        self.jail_candies = [CELL_NAME_TO_VALUE[v] for v in CELL_NAMES.values() if "_JAIL" in v ]
        self.bomb_candies = [CELL_NAME_TO_VALUE[v] for v in CELL_NAMES.values() if "_BOMB" in v ]
        self.chocolate = [CELL_NAME_TO_VALUE[v] for v in ["CONE_BASE"]]
        self.noncandies = [CELL_NAME_TO_VALUE[v] for v in ["STONE_BASE", "NA_BASE"]]


        self.cannot_move = [CELL_NAME_TO_VALUE[v] for v in
                            ["STONE_BASE", "BLACK_JAIL","BLUE_JAIL","BROWN_JAIL","GREEN_JAIL","PINK_JAIL","WHITE_JAIL","YELLOW_JAIL", "NA_BASE"]]
        self.special_candies = self.striped_candies_h + self.striped_candies_v1 + self.striped_candies_v2 + \
            self.flower_candies + self.snow_candies + self.tri_candies + self.chocolate

        self.game_board = None
        self.potential_start_coords = set()

    def get_score(self, candy_type):
        if candy_type in self.simple_candies:
            return 20
        elif candy_type in self.special_candies:
            return 40

        return 0

    def compute_explosions_chocolate(self, board, color):
        to_explode = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.candy_matches(board[i][j], color):
                    to_explode.append((i, j))

        return to_explode


    def isValidPosition(self, board, i, j):
        return  0 <= i < self.board_size and 0 <= j < self.board_size and board[i][j] != -1

    def get_flower_explosion(self, board, coords):
        return [
            getU(coords),
            getU(getU(coords)),
            getD(coords),
            getD(getD(coords)),
            getUL(coords),
            getUL(getUL(coords)),
            getUR(coords),
            getUR(getUR(coords)),
            getDL(coords),
            getDL(getDL(coords)),
            getDR(coords),
            getDR(getDR(coords)),
        ]

    def get_snow_explosion(self, board, p):
        return [
            getU(p),
            getU(getU(p)),
            getUL(p),
            getUL(getUL(p)),
            getDL(p),
            getDL(getDL(p)),
            getUR(p),
            getUR(getUR(p)),
            getD(p),
            getD(getD(p)),
            getDR(p),
            getDR(getDR(p)),

        ]

    def get_tri_explosion(self, board, p):
        return [
            getU(p),
            getUL(p),
            getU(getUL(p)),
            getDL(p),
            getD(getDL(p)),
            getD(p),
            getDR(p),
            getUR(p),
            getDR(getUR(p)),
        ]
    def get_stripe1_explosion(self, board, coords):
        to_explode = []
        candy_type = board[coords[0]][coords[1]]
        if candy_type in self.striped_candies_h:
            for k in range(self.board_size):
                to_explode.append((k, coords[1]))
        return to_explode

    def get_stripe2_explosion(self, board, coords):
        to_explode = []
        candy_type = board[coords[0]][coords[1]]
        ## explode right down
        if candy_type in self.striped_candies_v1:
            to_explode.append(coords)
            i, j = coords
            while 0 <= i < self.board_size and 0 <= j < self.board_size:
                if board[i][j] != -1:
                    to_explode.append((i, j))
                i, j = getUL((i, j))

            i, j = coords
            while 0 <= i < self.board_size and 0 <= j < self.board_size:
                if board[i][j] != -1:
                    to_explode.append((i, j))
                i, j = getDR((i, j))
        return to_explode

    def get_stripe3_explosion(self, board, coords):
        to_explode = []
        candy_type = board[coords[0]][coords[1]]
        ## explode right down
        if candy_type in self.striped_candies_v2:
            to_explode.append(coords)
            i, j = coords
            while 0 <= i < self.board_size and 0 <= j < self.board_size:
                if board[i][j] != -1:
                    to_explode.append((i, j))
                i, j = getUR((i, j))

            i, j = coords
            while 0 <= i < self.board_size and 0 <= j < self.board_size:
                if board[i][j] != -1:
                    to_explode.append((i, j))
                i, j = getDL((i, j))

        return to_explode

    def candy_matches(self, type1, type2):
        for match in self.match_list:
            if type1 in match and type2 in match:
                return True

        return False

    def compute_explosions_lines(self, board, start):
        ## indexing => even/odd, case, 0 or 1
        directions=[
                [[(-1, 0), (1, 0)],  # vertical
                 [(-1, -1), (0, 1)],
                 [(0, -1), (-1, 1)],],
                [[(-1, 0), (1, 0)],  # vertical
                  [(0, -1), (1, 1)],
                  [(1, -1), (0, 1)],]
        ]

        to_explode = set([])
        cum_open_list_counter = 0
        for dirs_index in range(3):
            open_list = [start]
            for d_index in range(2):
                i = start[0] + directions[start[1]%2][dirs_index][d_index][0]
                j = start[1] + directions[start[1]%2][dirs_index][d_index][1]
                while self.isValidPosition(board, i, j) \
                        and self.candy_matches(board[i][j], board[start[0]][start[1]]):
                    open_list.append((i, j))
                    move = (directions[j%2][dirs_index][d_index][0], directions[j%2][dirs_index][d_index][1])
                    i += move[0]
                    j += move[1]

            if len(open_list) >= 3:
                cum_open_list_counter += len(open_list)
                processed = set([])
                to_explode.update(set(open_list))
                while len(processed) != len(to_explode):
                    remaining = to_explode.difference(processed)
                    processed.update(to_explode)
                    for element in remaining:
                        if(self.isValidPosition(board, element[0], element[1])):
                            cell = board[element[0]][element[1]]
                            if cell in self.striped_candies_h:
                                to_explode.update(set(self.get_stripe1_explosion(board, element)))
                            elif cell in self.striped_candies_v1:
                                to_explode.update(set(self.get_stripe2_explosion(board, element)))
                            elif cell in self.striped_candies_v2:
                                to_explode.update(set(self.get_stripe3_explosion(board, element)))
                            elif cell in self.flower_candies:
                                to_explode.update(set(self.get_flower_explosion(board, element)))
                            elif cell in self.snow_candies:
                                to_explode.update(set(self.get_snow_explosion(board, element)))
                            elif cell in self.tri_candies:
                                to_explode.update(set(self.get_tri_explosion(board, element)))

            if len(open_list) >= 4 and board[start[0]][start[1]] != CELL_NAME_TO_VALUE["CONE_BASE"]:  # got special candy
                to_explode.remove(start)

        if DEBUG_EXPLOSIONS and cum_open_list_counter > 0:
            print "CumOpenListCounter : ", cum_open_list_counter, to_explode
        return to_explode, cum_open_list_counter

    def compute_explosions(self, start, end, board):
        chocolate_multiplier = 1
        to_explode = []
        line_explosion_size = 0
        if end and board[start[0]][start[1]] == CELL_NAME_TO_VALUE["CONE_BASE"]:  # chocolate
            to_explode = self.compute_explosions_chocolate(board, board[end[0]][end[1]])
        else:
            to_explode, line_explosion_size = self.compute_explosions_lines(board, start)

        to_explode = [v for v in to_explode if self.isValidPosition(board, v[0], v[1])]
        score = len(to_explode) * 10

        if DEBUG_EXPLOSIONS and to_explode:
            print "compute explosions : "
            print_board(board, to_explode)

        ## count bomb
        for (i, j) in to_explode:
            if board[i][j] in self.bomb_candies:
                score += 50

        ## remove stones nearby
        s1 = set(to_explode)
        orglength = len(s1)
        for pos in s1:
            if pos[1] % 2 == 0:
                dirs = [(-1, -1), (0, -1), (-1, 1), (0, 1), (-1, 0), (1, -1)]
            else:
                dirs = [(0, -1), (1, -1), (0, 1), (1, 1), (-1, 0), (1, 0)]
            for d in dirs:
                ni = pos[0] + d[0]
                nj = pos[1] + d[1]
                if self.isValidPosition(board, ni, nj) and board[ni][nj] == CELL_NAME_TO_VALUE["STONE_BASE"]:
                    to_explode.append((ni,nj))

        jail_cells = [ pos for pos in to_explode if (board[pos[0]][pos[1]] in self.jail_candies) ]
        jail_removal = len(jail_cells)

        for (i, j) in jail_cells:
            to_explode.remove((i, j))
            board[i][j] = CELL_NAME_TO_VALUE[CELL_NAMES[board[i][j]].replace("_JAIL", "_BASE")]
        score += 200 * jail_removal

        if line_explosion_size > 0:
            if(line_explosion_size == 4):
                score += 500
            elif line_explosion_size >= 5:
                score += 10000

        #if len(to_explode) > 0:
        #    print '\n\nStarting board:'
        #    dbg.print_board(board)
        s2 = sorted(list(set(to_explode)))
        afterstone = len(s2)

        nStone = (afterstone - orglength)
        score += (150 * nStone)
        # Slide the other candies down after explosions take place'

        for coord in s2:
            i, j = coord
            while i > 0:
                if board[i-1][j] != -1 and (i-1, j) not in self.potential_start_coords:
                    self.potential_start_coords.add((i, j))
                ## don't slide down if jail
                if(board[i-1][j] in self.jail_candies): break
                if(board[i-1][j] == CELL_NAME_TO_VALUE["STONE_BASE"]): break

                board[i][j], board[i-1][j] = board[i-1][j], board[i][j]
                i -= 1
            board[i][j] = -1

        if DEBUG_EXPLOSIONS and to_explode:
            print "line explosion size : ", line_explosion_size, " stone : ", nStone, " jail: ", jail_removal
            print "after slide down : *potential start*"
            print_board(board, self.potential_start_coords)


        #if len(to_explode) > 0:
           # print '\nResult from {0}, count={1}, score={2}:'.format(start, len(to_explode), score)
            #dbg.print_board(board)

        return score, board

    def evaluate_board(self, start, end, board):
        total_score, new_board = self.compute_explosions(start, end, board)
        score = total_score
        multiplier = 1
        while score > 0:
            use_new = False
            if use_new:
                potential_start = deepcopy(self.potential_start_coords)
                self.potential_start_coords = set()
                score = 0
                for coord in potential_start:
                    score, new_board = self.compute_explosions((coord[0], coord[1]), None, new_board)
                    if score > 0:
                        total_score += (score * multiplier)
                        multiplier += 2
            else:
                for i in range(0, self.board_size):
                    for j in range(0, self.board_size):
                        score, new_board = self.compute_explosions((i, j), None, new_board)
                        if score > 0:
                            total_score += (score * multiplier)
                            multiplier += 2

        return total_score, new_board


    def check_direction(self, start, dir):
            end = (start[0]+dir[0], start[1]+dir[1])
            board = deepcopy(self.game_board)
            # invalid position
            if start[0] < 0 or start[0] > self.board_size or end[0] < 0 or end[0] > self.board_size\
                    or start[1] < 0 or start[1] > self.board_size or end[1] < 0 or end[1] > self.board_size:
                return -1, [], None

            # cannot move
            if board[start[0]][start[1]] in self.cannot_move or board[end[0]][end[1]] in self.cannot_move:
                return -1, [], None
            # swap
            board[start[0]][start[1]], board[end[0]][end[1]] = board[end[0]][end[1]], board[start[0]][start[1]]
            score_start, start_board = self.evaluate_board(start, end, board)
            score_end, end_board = self.evaluate_board(end, start, board)

            if score_start > score_end:
                return score_start, [start, end], start_board
            else:
                return score_end, [end, start], end_board

    def getPossibleDirections(self, j):
        if(j%2 == 0):
            possible_directions = [(1, 0), (-1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]
        else:
            possible_directions = [(1, 0), (-1, 0), (1, -1), (1, 1), (0, 1), (0, -1)]
        return possible_directions

    def solve_board(self, board):
        self.game_board = board
        max_score = 0
        chosen_move = []
        for i in range(0, 8):
            for j in range(0, 8):
                for d in self.getPossibleDirections(j):
                    score, move, b = self.check_direction((i, j), d)
                    if score >= max_score:
                        max_score = score
                        chosen_move = move

        return max_score, chosen_move

