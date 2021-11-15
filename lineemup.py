# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python
import random
import time
import numpy as np
import traceback
import sys

ComplexCounter = 0
SimpleCounter = 0
DepthList = [0]
totalDepths = []
winnerTracker = []
gameCounterTracker = 0
averageTimeTracker = []
totalStatesTracker = []
totalDepthsTracker = []
totalAverageDepthTracker = []
totalARDTracker = []
totalMovesTracker = []
class Game:
    MINIMAX = 0
    ALPHABETA = 1
    HUMAN = 2
    AI = 3
    SIMPLE_EVAL = 4
    COMPLEX_EVAL = 5

    def __init__(self, n, b, barray, s, d1, d2, a1, a2, ai_timeout, recommend=True):
        self.n = n
        self.b = b
        self.barray = barray

        self.s = s
        self.d1 = d1
        self.d2 = d2
        self.a1 = a1
        self.a2 = a2
        self.ai_timeout = ai_timeout
        self.recommend = recommend

        self.initialize_game()

    def initialize_game(self):
        self.current_state = np.full((self.n, self.n), '.')

        for x in self.barray:
            self.current_state[x[0]][x[1]] = '~'

        self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][0:self.n]

        # Player X always plays first
        self.player_turn = 'X'

        # Determine diagonal positions to check
        self.diagonal_starting_positions = []
        self.other_diagonal_starting_positions = []
        for y in range(self.n - self.s + 1):
            self.diagonal_starting_positions.append([y, 0])
            if (y != 0):
                self.diagonal_starting_positions.append([0, y])

        x = 0
        for y in range(self.n - 1, self.s - 2, -1):
            self.other_diagonal_starting_positions.append([y, 0])
            if (y != self.n - 1):
                self.other_diagonal_starting_positions.append([self.n - 1, x])
            x = x + 1
    def writeBoardToFile(self, file):
        # Print column headers (from the alphabet)
        # Build row separator
        rowsep = '-----'
        file.write("\n \n Drawing Current Board: \n \n" )
        file.write('\n    |  ')
        for x in range(self.n):
            file.write(self.alphabet[x] + "  |  ")
            rowsep = rowsep + '------'

        file.write(F"\n{rowsep}\n")

        for y in range(self.n):
            # Print row headers (0-n)
            file.write(str(y))

            # Print the actual contents of the board
            p = self.current_state[0][y].replace('.', ' ')
            file.write(F'   |  {p}  | ')

            for x in range(1, self.n):
                p = self.current_state[x][y].replace('.', ' ') # In code, we have a '.' representing a blank square, however, we can just show a blank square to the user

                # Print the actual contents of the board
                file.write(F'{p}  | ')

            file.write(F"\n{rowsep}\n")

        file.write("\n")

    def draw_board(self):
        # Print column headers (from the alphabet)
        # Build row separator
        rowsep = '-----'
        print('\n    |  ', end='')
        for x in range(self.n):
            print(self.alphabet[x] + "  |  ", end='')
            rowsep = rowsep + '------'

        print(F"\n{rowsep}")

        for y in range(self.n):
            # Print row headers (0-n)
            print(y, end='')

            # Print the actual contents of the board
            p = self.current_state[0][y].replace('.', ' ')
            print(F'   |  {p}  | ', end=" ")

            for x in range(1, self.n):
                p = self.current_state[x][y].replace('.', ' ') # In code, we have a '.' representing a blank square, however, we can just show a blank square to the user

                # Print the actual contents of the board
                print(F'{p}  | ', end=" ")

            print(F"\n{rowsep}")

        print()

    def is_valid(self, px, py):
        ipx = -1
        for x in range(len(self.alphabet)):
            if (self.alphabet[x] == px):
                ipx = x
                break

        if (ipx == -1):
            return (0, 0, False)

        if (not py.isdigit()):
            return (0, 0, False)

        py = int(py)
        if ipx < 0 or ipx > self.n-1 or py < 0 or py > self.n-1:
            return (0, 0, False)
        elif self.current_state[ipx][py] != '.':
            return (0, 0, False)
        else:
            return (ipx, py, True)

    def is_end(self):
        # Vertical win
        for i in range(0, self.n):
            l = 0
            di = -1
            dj = -1
            for j in range(0, self.n):
                if (l == 0 and (self.current_state[j][i] != '.' and self.current_state[j][i] != '~')):
                    di = i
                    dj = j
                    l += 1
                elif ((di > -1 and dj > -1) and (self.current_state[dj][di] == self.current_state[j][i])):
                    l += 1
                elif (self.current_state[j][i] != '.' and self.current_state[j][i] != '~'):
                    di = i
                    dj = j
                    l = 1
                else:
                    l = 0

                if (l == self.s):
                    return self.current_state[dj][di]

        # Horizontal win
        for i in range(0, self.n):
            l = 0
            di = -1
            dj = -1
            for j in range(0, self.n):
                if (l == 0 and (self.current_state[i][j] != '.' and self.current_state[i][j] != '~')):
                    di = i
                    dj = j
                    l += 1
                elif ((di > -1 and dj > -1) and (self.current_state[di][dj] == self.current_state[i][j])):
                    l += 1
                elif (self.current_state[i][j] != '.' and self.current_state[i][j] != '~'):
                    di = i
                    dj = j
                    l = 1
                else:
                    l = 0

                if (l == self.s):
                    return self.current_state[di][dj]

        # Diagonal win (from left to right)
        for d in range(len(self.diagonal_starting_positions)):
            st_x = self.diagonal_starting_positions[d][0]
            st_y = self.diagonal_starting_positions[d][1]
            l = 0
            di = -1
            dj = -1
            while (st_x < self.n and st_y < self.n):
                if (l == 0 and (self.current_state[st_x][st_y] != '.' and self.current_state[st_x][st_y] != '~')):
                    di = st_x
                    dj = st_y
                    l += 1
                elif ((di > -1 and dj > -1) and (self.current_state[di][dj] == self.current_state[st_x][st_y])):
                    l += 1
                elif (self.current_state[st_x][st_y] != '.' and self.current_state[st_x][st_y] != '~'):
                    di = st_x
                    dj = st_y
                    l = 1
                else:
                    l = 0

                st_x += 1
                st_y += 1

                if (l == self.s):
                    return self.current_state[di][dj]

        # Diagonal win (from right to left)
        for d in range(len(self.other_diagonal_starting_positions)):
            st_x = self.other_diagonal_starting_positions[d][0]
            st_y = self.other_diagonal_starting_positions[d][1]
            l = 0
            di = -1
            dj = -1
            while (st_x > -1 and st_y < self.n):
                if (l == 0 and (self.current_state[st_x][st_y] != '.' and self.current_state[st_x][st_y] != '~')):
                    di = st_x
                    dj = st_y
                    l += 1
                elif ((di > -1 and dj > -1) and (self.current_state[di][dj] == self.current_state[st_x][st_y])):
                    l += 1
                elif (self.current_state[st_x][st_y] != '.' and self.current_state[st_x][st_y] != '~'):
                    di = st_x
                    dj = st_y
                    l = 1
                else:
                    l = 0

                st_x = st_x - 1
                st_y = st_y + 1

                if (l == self.s):
                    return self.current_state[di][dj]

        # Is whole board full?
        for i in range(0, self.n):
            for j in range(0, self.n):
                # There's an empty field, we continue the game
                if (self.current_state[i][j] == '.'):
                    return None

        # It's a tie!
        return '.'

    def check_end(self):
        self.result = self.is_end()

        # Printing the appropriate message if the game has ended
        if self.result != None:
            if self.result == 'X':
                print('The winner is X!')
            elif self.result == 'O':
                print('The winner is O!')
            elif self.result == '.':
                print("It's a tie!")

            self.initialize_game()

        return self.result

    def input_move(self):
        print(F'Player {self.player_turn}, enter your move:')

        while True:
            px = input(F'  Enter the column ({self.alphabet[0]} - {self.alphabet[-1]}): ')
            py = input(F'  Enter the row (0 - {self.n - 1}): ')

            rpx, rpy, isv = self.is_valid(px, py)
            if (isv):
                return rpx, rpy

            print('\n  ERROR: The move is not valid! Try again.\n')

    def switch_player(self):
        if self.player_turn == 'X':
            self.player_turn = 'O'
        elif self.player_turn == 'O':
            self.player_turn = 'X'

        return self.player_turn

    def minimax(self, depth, startDepth, start_time, eval_method, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to inf or -inf as worse than the worst case:
        value = np.inf
        if max:
            value = -np.inf

        x = None
        y = None
        ARD = 0
        Children = 0
        result = self.is_end()
        elapsed_t = time.time() - start_time
        global DepthList
        if depth == 0 or result != None or elapsed_t >= self.ai_timeout-.2:
            if eval_method == self.SIMPLE_EVAL:
                global SimpleCounter
                SimpleCounter += 1
                DepthList[depth] = DepthList[depth]+1
                return (self.simple_heuristic(), x, y, startDepth-depth)
            elif eval_method == self.COMPLEX_EVAL:
                global ComplexCounter
                ComplexCounter += 1
                DepthList[depth] = DepthList[depth]+1
                return (self.complex_heuristic(), x, y, startDepth-depth)

        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _, childARD) = self.minimax(depth-1,startDepth, start_time, eval_method, max=False)
                        if v > value:
                            ARD += childARD
                            Children += 1
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _, childARD) = self.minimax(depth-1,startDepth, start_time, eval_method, max=True)
                        if v < value:
                            ARD += childARD
                            Children += 1
                            value = v
                            x = i
                            y = j

                    self.current_state[i][j] = '.'
        return (value, x, y, ARD/Children)

    def alphabeta(self, depth, startDepth, start_time, eval_method, alpha=-np.inf, beta=np.inf, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to inf or -inf as worse than the worst case:
        value = np.inf
        if max:
            value = -np.inf

        x = None
        y = None
        ARD = 0
        Children = 0

        result = self.is_end()
        elapsed_t = time.time() - start_time
        global DepthList
        if depth == 0 or result != None or elapsed_t >= self.ai_timeout-.2:
            if eval_method == self.SIMPLE_EVAL:
                global SimpleCounter
                SimpleCounter += 1
                DepthList[depth] = DepthList[depth]+1
                return (self.simple_heuristic(), x, y, startDepth-depth)
            elif eval_method == self.COMPLEX_EVAL:
                global ComplexCounter
                ComplexCounter +=1
                DepthList[depth] = DepthList[depth]+1
                return (self.complex_heuristic(), x, y, startDepth-depth)

        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _, childARD) = self.alphabeta(depth-1, startDepth, start_time, eval_method, alpha, beta, max=False)
                        ARD += childARD
                        Children += 1
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _, childARD) = self.alphabeta(depth-1, startDepth, start_time, eval_method, alpha, beta, max=True)
                        if v < value:
                            ARD+= childARD
                            Children += 1
                            value = v
                            x = i
                            y = j

                    self.current_state[i][j] = '.'

                    if max:
                        if value >= beta:
                            return (value, x, y, ARD/Children)
                        if value > alpha:
                            alpha = value
                    else:
                        if value <= alpha:
                            return (value, x, y, ARD/Children)
                        if value < beta:
                            beta = value

        return (value, x, y, ARD/Children)

    # Simple heuristic
    # The idea is to check all rows/columns/diagonals
    # It will add 2 to the score if we see our piece in a given position
    # and subtract 2 to the score if we see an opponents piece
    def simple_heuristic(self):
        score = 0

        # Rows
        for y in range(self.n):
            s = 0
            for x in range(self.n):
                s = self.simple_heuristic_evaluator(x, y, s)
            score = score + s

        # Columns
        for x in range(self.n):
            s = 0
            for y in range(self.n):
                s = self.simple_heuristic_evaluator(x, y, s)
            score = score + s

        # Diagonals (left to right)
        for d in range(len(self.diagonal_starting_positions)):
            st_x = self.diagonal_starting_positions[d][0]
            st_y = self.diagonal_starting_positions[d][1]
            s = 0
            while (st_x < self.n and st_y < self.n):
                s = self.simple_heuristic_evaluator(st_x, st_y, s)
                st_x = st_x + 1
                st_y = st_y + 1
            score = score + s

        # Diagonals (right to left)
        for d in range(len(self.other_diagonal_starting_positions)):
            st_x = self.diagonal_starting_positions[d][0]
            st_y = self.diagonal_starting_positions[d][1]
            s = 0
            while (st_x > -1 and st_y < self.n):
                s = self.simple_heuristic_evaluator(st_x, st_y, s)
                st_x = st_x - 1
                st_y = st_y + 1
            score = score + s

        return score

    def simple_heuristic_evaluator(self, x, y, s):
        if (self.current_state[x][y] == 'O'):
            s = s + 2
        elif (self.current_state[x][y] == 'X'):
            s = s - 2

        return s

    # Complex heuristic
    # The idea is to check all rows/columns/diagonals
    # If we have a possible winning line of length s filled with empty tiles and at least one of our piece (I call this a psuedo win)
    # we add/substract to a total score. The score we add/subtract will depend on how many empty tiles are left
    def complex_heuristic(self):
        score = 0

        # Rows
        for rows in range(self.n):
            dx = 0
            while (dx < self.n):
                s, dx, _ = self.complex_heuristic_evaluator(dx, rows, "row")
                score = score + s

        # Columns
        for columns in range(self.n):
            dy = 0
            while (dy < self.n):
                s, _, dy = self.complex_heuristic_evaluator(columns, dy, "column")
                score = score + s

        # Diagonal (left to right)
        for i in range(len(self.diagonal_starting_positions)):
            dx = self.diagonal_starting_positions[i][0]
            dy = self.diagonal_starting_positions[i][1]
            while (dx < self.n and dy < self.n):
                s, dx, dy = self.complex_heuristic_evaluator(dx, dy, "diagonal-ltr")
                score = score + s

        # Diagonal (right to left)
        for i in range(len(self.other_diagonal_starting_positions)):
            dx = self.other_diagonal_starting_positions[i][0]
            dy = self.other_diagonal_starting_positions[i][1]
            while (dx > -1 and dy < self.n):
                s, dx, dy = self.complex_heuristic_evaluator(dx, dy, "diagonal-rtl")
                score = score + s

        return score

    # Evaluates a given row/column/diagonal
    def complex_heuristic_evaluator(self, x, y, t):
        di = x
        dj = y
        fnws = 0
        bnws = 0

        # Determine movement vectors (dx, dy)
        if (t == "row"):
            dx = 1
            dy = 0
        elif (t == "column"):
            dx = 0
            dy = 1
        elif (t == "diagonal-ltr"):
            dx = 1
            dy = 1
        else: # diagonal-rtl
            dx = -1
            dy = 1

        # Starting on an empty space can yield an advantage to either O or X. Determine who has the advantage by seeing what the next piece is
        while (((di > -1 and di < self.n) and dj < self.n) and (self.current_state[di][dj] == '.' or self.current_state[di][dj] == '~')):
            if (self.current_state[di][dj] == '.'):
                if (bnws < self.s):
                    bnws = bnws + 1
            else:
                bnws = 0

            di = di + dx
            dj = dj + dy

        # The whole row/column/diagonal did not have an X or O piece
        if ((di <= -1 or di >= self.n) or dj >= self.n):
            return (0, di, dj)

        s = 0
        cs = 0
        e = 0
        my_piece = self.current_state[di][dj]
        if (my_piece == 'O'):
            e = 1
        elif (my_piece == 'X'):
            e = -1

        while ((di > -1 and di < self.n) and dj < self.n):
            if (self.current_state[di][dj] == my_piece):
                if (bnws > 0 and (fnws + bnws + cs >= self.s)):
                    bnws = bnws - 1
                cs = cs + 1
                if (s == 0):
                    s = 2
            elif (self.current_state[di][dj] == '.'):
                fnws = fnws + 1
                if (bnws > 0 and (fnws + bnws + cs >= self.s)):
                    bnws = bnws - 1
            else:
                cs = 0
                # Psuedo win
                if (fnws + bnws + cs >= self.s):
                    n = self.s - (fnws + bnws)
                    w = abs(n) ** 3
                    s = s + w

                    if (fnws + bnws == 0):
                        s = s + 100000000000
                    #elif (fnws + bnws == 1):
                    #s = s + 10000
                    #elif (fnws + bnws == 2):
                    #s = s + 1000

                    di = di - dx
                    dj = dj - dy

                break

            # Psuedo win
            if (fnws + bnws + cs >= self.s):
                n = self.s - (fnws + bnws)
                w = abs(n) ** 3
                s = s + w

                if (fnws + bnws == 0):
                    s = s + 100000000000
                    break
                #elif (fnws + bnws == 1):
                #s = s + 10000
                #elif (fnws + bnws == 2):
                #s = s + 1000

            di = di + dx
            dj = dj + dy

        if (e < 0 and s != 0):
            s = -s

        return (s, di, dj)

    def play(self, px_algo=None, po_algo=None, player_x=None, player_o=None, px_eval=None, po_eval=None):
        if px_algo == None:
            px_algo = self.ALPHABETA
        if po_algo == None:
            po_algo = self.ALPHABETA
        if player_x == None:
            player_x = self.HUMAN
        if px_eval == None:
            px_eval = self.SIMPLE_EVAL
        if player_o == None:
            player_o = self.HUMAN
        if po_eval == None:
            po_eval == self.SIMPLE_EVAL
        fileName = "gametrace" + str(self.n) + str(self.b) + str(self.s) + str(self.ai_timeout) + ".txt"
        f = open(fileName, "w")

        f.write("The board size is: " + str(self.n) + "\n")
        f.write("The number of blocks are: " + str(self.b) + "\n")
        f.write("The winning connection length is: " + str(self.s) + "\n")
        f.write("The AI timeout is: " + str(self.ai_timeout) + "\n")
        f.write("The blocks are positioned at: " + str(self.barray) + "\n")
        if (player_x == self.HUMAN):
            f.write("Player 1 is a human\n")
        else:
            f.write("Player 1 is an AI\n")
            f.write("The maximum depth of the adversarial search for player 1 is: " + str(self.d1) + "\n")
            if (px_algo== self.MINIMAX):
                f.write("Player 1 uses minimax\n")
            else:
                f.write("Player 1 uses alpha-beta\n")
            if(px_eval == 5):
                f.write("Player 1 uses the complex heuristic function \n")
            else:
                f.write("Player 1 uses the simple heuristic function \n")

        if (player_o == self.HUMAN):
            f.write("Player 2 is a human\n")
        else:
            f.write("Player 2 is an AI\n")
            f.write("The maximum depth of the adversarial search for player 2 is: " + str(self.d2) + "\n")
            if (po_algo == self.MINIMAX):
                f.write("Player 2 uses minimax\n")
            else:
                f.write("Player 2 uses alpha-beta\n")
            if(po_eval == 5):
                f.write("Player 2 uses the complex heuristic function \n")
            else:
                f.write("Player 2 uses the simple heuristic function \n")

        self.writeBoardToFile(f)

        #self.complex_heuristic()
        #return
        global totalDepths
        totalDepths = []
        global totalDepthsTracker
        if totalDepthsTracker == [] :
            if(self.d2 > self.d1):
                for ddd in range (0, self.d2):
                    totalDepthsTracker.append(0)
            else:
                for ddd in range (0, self.d1):
                    totalDepthsTracker.append(0)

        if(self.d2 > self.d1):
            for ddd in range (0, self.d2):
                totalDepths.append(0)
        else:
            for ddd in range (0, self.d1):
                totalDepths.append(0)

        print(str(len(totalDepths)))
        global ComplexCounter
        global SimpleCounter
        global DepthList
        global winnerTracker
        global gameCounterTracker
        global averageTimeTracker
        global totalStatesTracker
        global totalAverageDepthTracker
        global totalARDTracker
        global totalMovesTracker
        averageHeuristic = []
        totalStatesEvaluated = 0
        averageAverageDepth = []


        averageARD = []
        moveCounter = 0
        while True:
            self.draw_board()
            if self.check_end():
                gameCounterTracker += 1
                if self.result != None:
                    if self.result == 'X':
                        f.write('\nThe winner is X!\n')
                        if(px_eval == 4):
                            winnerTracker.append('e1')
                        if(px_eval == 5):
                            winnerTracker.append('e2')
                    elif self.result == 'O':
                        f.write('\nThe winner is O!\n')
                        if(po_eval == 4):
                            winnerTracker.append('e1')
                        if(po_eval == 5):
                            winnerTracker.append('e2')

                    elif self.result == '.':
                        f.write("\nIt's a tie!\n")
                        winnerTracker.append('.')

                f.write("\n\nThe average time taken per heuristic was: " + str(sum(averageHeuristic)/len(averageHeuristic)) + "\n")
                averageTimeTracker.append(sum(averageHeuristic)/len(averageHeuristic))
                f.write("The total number of states evaluated was: " + str(totalStatesEvaluated) + "\n")
                totalStatesTracker.append(totalStatesEvaluated)
                f.write("The average of the average depths was: " + str(sum(averageAverageDepth)/len(averageAverageDepth)) + "\n")
                totalAverageDepthTracker.append(sum(averageAverageDepth)/len(averageAverageDepth))
                if(self.d1 > self.d2):
                    for dd in range (0, self.d1):
                        f.write("Total states evaluated at depth " + str(dd+1) + ": " + str(totalDepths[dd]) + "\n")
                        totalDepthsTracker[dd] += totalDepths[dd]
                else:
                    for dd in range (0, self.d2):
                        f.write("Total states evaluated at depth " + str(dd+1) + ": " + str(totalDepths[dd]) + "\n")
                        totalDepthsTracker[dd] += totalDepths[dd]
                f.write("The average ARD of the moves taken in the game was: " + str(sum(averageARD)/len(averageARD)) + "\n")
                totalARDTracker.append(sum(averageARD)/len(averageARD))
                f.write("The total number of moves taken in the game was: " + str(moveCounter) + "\n\n")
                totalMovesTracker.append(moveCounter)
                return

            ComplexCounter = 0
            SimpleCounter = 0
            averageDepth = 0
            start = time.time()

            if self.player_turn == 'X':
                if (px_algo == self.MINIMAX):
                     if ((player_x == self.HUMAN and self.recommend == True) or (player_x == self.AI)):
                        DepthList = [0] * self.d1
                        (_, x, y, ARD) = self.minimax(self.d1, self.d1,start, px_eval, max=False)
                else:
                    if ((player_x == self.HUMAN and self.recommend == True) or (player_x == self.AI)):
                        DepthList = [0] * self.d1
                        (m, x, y, ARD) = self.alphabeta(self.d1, self.d1, start, px_eval, max=False)
            else:
                if (po_algo == self.MINIMAX):
                    if ((player_o == self.HUMAN and self.recommend == True) or (player_o == self.AI)):
                        DepthList = [0] * self.d2
                        (_, x, y, ARD) = self.minimax(self.d2, self.d2, start, po_eval, max=True)
                else:
                    if ((player_o == self.HUMAN and self.recommend == True) or (player_o == self.AI)):
                        DepthList = [0] * self.d2
                        (m, x, y, ARD) = self.alphabeta(self.d2, self.d2, start, po_eval, max=True)


            moveCounter += 1
            end = time.time()

            if (player_o == self.AI and (end - start) > self.ai_timeout) or (player_x == self.AI and (end - start) > self.ai_timeout):
                if self.player_turn == 'X':
                    self.result = 'O'
                else:
                    self.result = 'X'
                print(F"Player {('X' if self.player_turn == 'X' else 'O')} has taken too long to make a move.")
                print(F"The winner is {('O' if self.player_turn == 'X' else 'X')}!")
                f.write(F"Player {('X' if self.player_turn == 'X' else 'O')} has taken too long to make a move.\n")
                f.write(F"The winner is {('O' if self.player_turn == 'X' else 'X')}!\n")

                if self.result == 'X':
                    if(px_eval == 4):
                        winnerTracker.append('e1')
                    if(px_eval == 5):
                        winnerTracker.append('e2')
                elif self.result == 'O':
                    f.write('\nThe winner is O!\n')
                    if(po_eval == 4):
                        winnerTracker.append('e1')
                    if(po_eval == 5):
                        winnerTracker.append('e2')
                elif self.result == '.':
                    f.write("\nIt's a tie!\n")
                    winnerTracker.append('.')
                if(len(averageHeuristic) != 0):
                    f.write("\n\nThe average time taken per heuristic was: " + str(sum(averageHeuristic)/len(averageHeuristic)) + "\n")
                    averageTimeTracker.append(sum(averageHeuristic)/len(averageHeuristic))
                f.write("The total number of states evaluated was: " + str(totalStatesEvaluated) + "\n")
                totalStatesTracker.append(totalStatesEvaluated)
                if(len(averageAverageDepth) != 0):
                    f.write("The average of the average depths was: " + str(sum(averageAverageDepth)/len(averageAverageDepth)) + "\n")
                    totalAverageDepthTracker.append(sum(averageAverageDepth)/len(averageAverageDepth))
                if(self.d1 > self.d2):
                    for dd in range (0, self.d1):
                        f.write("Total states evaluated at depth " + str(dd+1) + ": " + str(totalDepths[dd]) + "\n")
                        totalDepthsTracker[dd] += totalDepths[dd]
                else:
                    for dd in range (0, self.d2):
                        f.write("Total states evaluated at depth " + str(dd+1) + ": " + str(totalDepths[dd]) + "\n")
                        totalDepthsTracker[dd] += totalDepths[dd]
                if(len(averageARD) != 0):
                    f.write("The average ARD of the moves taken in the game was: " + str(sum(averageARD)/len(averageARD)) + "\n")
                    totalARDTracker.append(sum(averageARD)/len(averageARD))
                f.write("The total number of moves taken in the game was: " + str(moveCounter) + "\n\n")
                totalMovesTracker.append(moveCounter)
                gameCounterTracker += 1
                return

            if (self.player_turn == 'X' and player_x == self.HUMAN) or (self.player_turn == 'O' and player_o == self.HUMAN):
                if self.recommend:

                    print(F'Evaluation time: {round(end - start, 7)}s')
                    print(F'Recommended move: {self.alphabet[x]} {y}')
                    f.write(F'Evaluation time: {round(end - start, 7)}s\n')
                    f.write(F'Recommended move: {self.alphabet[x]} {y}\n')

                (x, y) = self.input_move()
            if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
                averageHeuristic.append(round(end - start, 7))
                print(F'Evaluation time: {round(end - start, 7)}s')
                print(F'Player {self.player_turn} under AI control plays: {self.alphabet[x]} {y}')
                f.write(F'\n\nPlayer {self.player_turn} under AI control plays: {self.alphabet[x]} {y}\n')



            self.current_state[x][y] = self.player_turn
            self.writeBoardToFile(f)
            f.write(F'Evaluation time: {round(end - start, 7)}s \n')



            if(ComplexCounter != 0):
                 totalStatesEvaluated += ComplexCounter
                 f.write(str(ComplexCounter) + " states were evaluated by the heuristic function\n")
            if(SimpleCounter != 0):
                 totalStatesEvaluated += SimpleCounter
                 f.write(str(SimpleCounter) + " states were evaluated by the heuristic function\n")

            if self.player_turn == 'X':
                for z in range (0, self.d1):
                    totalDepths[self.d1-1-z] += DepthList[z]
                    f.write("States evaluated at depth " + str(self.d1-z) + ": " + str(DepthList[z]) + "\n")
                    averageDepth += DepthList[z]*(self.d1-z)
                averageDepth = averageDepth/sum(DepthList)
                averageAverageDepth.append(averageDepth)
                f.write("This gives an average depth of: " + str(averageDepth) + "\n")
            if self.player_turn == 'O':
                for z in range (0, self.d2):
                    totalDepths[self.d2-1-z] += DepthList[z]
                    f.write("States evaluated at depth " + str(self.d2-z) + ": " + str(DepthList[z]) + "\n")
                    averageDepth += DepthList[z]*(self.d2-z)
                averageDepth = averageDepth/sum(DepthList)
                averageAverageDepth.append(averageDepth)
                f.write("This gives an average depth of: " + str(averageDepth) + "\n")
            f.write("The average recursion depth is: " + str(ARD) + "\n")
            averageARD.append(ARD)
            self.switch_player()

        f.close

class GameBuilder:
    def build_game(config_path):
        try:
            with open(config_path, "r") as config:
                for line in config:
                    if (line.split('=')[0] == "boardSize"):
                        board_size = int(line.split('=')[1])
                    elif (line.split('=')[0] == "blockCount"):
                        block_count = int(line.split('=')[1])
                    elif (line.split('=')[0] == "blockArray"):
                        if(line.split('=')[1] == 'r\n'):
                            randomBlockArray = True
                        else:
                            temp = iter(line.split('=')[1].split())
                            block_array = [(int(ele.split(",")[0]), int(ele.split(",")[1])) for ele in temp]
                            randomBlockArray = False
                    elif (line.split('=')[0] == "winLength"):
                        win_length = int(line.split('=')[1])
                    elif (line.split('=')[0] == "maxDepthD1"):
                        max_depthD1 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "maxDepthD2"):
                        max_depthD2 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "aiTimeout"):
                        ai_timeout = int(line.split('=')[1])
                    elif (line.split('=')[0] == "alphabeta1"):
                        alphabeta1 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "alphabeta2"):
                        alphabeta2 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "p1"):
                        p1 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "p2"):
                        p2 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "p1Eval"):
                        p1_eval = int(line.split('=')[1])
                    elif (line.split('=')[0] == "p2Eval"):
                        p2_eval = int(line.split('=')[1])

                config.close()
        except Exception:
            print("ERROR: Could not open file", config_path)
            print(traceback.format_exc())
            return (None, None, None, None, None, None, None)


        # Size of board
        if (board_size < 3 or board_size > 10):
            print("ERROR: Invalid game configuration. Board size must be between 3 and 10 inclusive")
            return (None, None, None, None, None, None, None)

        # Number of blocks
        if (block_count < 0 or block_count > 2 * board_size):
            print(F"ERROR: Invalid game configuration. Number of blocks placed must be between 0 and {2 * board_size} inclusive")
            return (None, None, None, None, None, None, None)

        if (randomBlockArray):
            block_array = []

            for _ in range(0, block_count):
                blockfound = False
                while(not blockfound):
                    randomX = random.randint(0,board_size-1)
                    randomY = random.randint(0,board_size-1)
                    randomXY = [randomX, randomY]
                    if(block_array.__contains__(randomXY)):
                        continue
                    else:
                        block_array.append(randomXY)
                        blockfound = True


        # Number of blocks
        if (len(block_array) != block_count):
            print("ERROR: Invalid game configuration. Number of blocks does not match size of block array in config file")
            return (None, None, None, None, None, None, None)



        # Validation for the blocks placed
        for x in range(len(block_array)):
            block_placed = block_array[x]
            if (block_placed[0] >= board_size or block_placed[1] >= board_size):
                print(F"ERROR: Invalid game configuration. Block {block_placed} not placed inside board")
                return (None, None, None, None, None, None, None)

        # Winning line up size
        if (win_length < 3 or win_length > board_size):
            print(F"ERROR: Invalid game configuration. Winning line up size must be between 3 and {board_size} inclusive")
            return (None, None, None, None, None, None, None)


        return (alphabeta1, alphabeta2, p1, p2, p1_eval, p2_eval, Game(board_size, block_count, block_array, win_length, max_depthD1, max_depthD2, alphabeta1, alphabeta2, ai_timeout, recommend=True))

def playrtimes(r, game_config):
    ax, ao,  px, po, px_e, po_e, g = GameBuilder.build_game(game_config)
    global winnerTracker
    winnerTracker = []
    global gameCounterTracker
    gameCounterTracker = 0
    global averageTimeTracker
    averageTimeTracker = []
    global totalStatesTracker
    totalStatesTracker = []
    global totalDepthsTracker
    totalDepthsTracker = []
    global totalAverageDepthTracker
    totalAverageDepthTracker = []
    global totalARDTracker
    totalARDTracker = []
    global totalMovesTracker
    totalMovesTracker = []
    scorefileName = "Scoreboard" + str(g.n) + str(g.b) + str(g.s) + str(g.ai_timeout) + ".txt"
    scoreFile = open(scorefileName, "w")
    scoreFile.write("Gameboard size n: " + str(g.n) + "\n")
    scoreFile.write("Number of blocks b: " + str(g.b) + "\n")
    scoreFile.write("Winning connection length s: " + str(g.s) + "\n")
    scoreFile.write("AI timeout t: " + str(g.ai_timeout) + "\n")
    scoreFile.write("Player 1 depth: " + str(g.d1) + "\n")
    if(g.a1 == 1):
        scoreFile.write("Player 1 algo: ALPHABETA\n")
    else:
        scoreFile.write("Player 1 algo: Minimax\n")
    if(px_e == 5):
        scoreFile.write("Player 1 heuristic: Complex \n")
    else:
        scoreFile.write("Player 1 heuristic: Simple \n")
    scoreFile.write("Player 2 depth: " + str(g.d2) + "\n")
    if(g.a2 == 1):
        scoreFile.write("Player 2 algo: ALPHABETA\n")
    else:
        scoreFile.write("Player 2 algo: Minimax\n")
    if(po_e == 5):
        scoreFile.write("Player 2 heuristic: Complex \n")
    else:
        scoreFile.write("Player 2 heuristic: Simple \n")

    if (g != None):
        for _ in range (r):
            ax, ao,  px, po, px_e, po_e, g = GameBuilder.build_game(game_config)
            g.play(px_algo = ax, po_algo = ao , player_x=px, player_o=po, px_eval=px_e, po_eval=po_e)


    # switchDepthHelper = g.d1
    # g.d1 = g.d2
    # g.d2 = switchDepthHelper
    # switchAlphaBetaHelper = g.a1
    # g.a1 = g.a2
    # g .a2 = switchAlphaBetaHelper

    if (g != None):
        for _ in range (r):
            ax, ao,  px, po, px_e, po_e, g = GameBuilder.build_game(game_config)
            switchDepthHelper = g.d1
            g.d1 = g.d2
            g.d2 = switchDepthHelper
            switchAlphaBetaHelper = g.a1
            g.a1 = g.a2
            g.a2 = switchAlphaBetaHelper
            g.play(px_algo = ao, po_algo = ax , player_x=po, player_o=px, px_eval=po_e, po_eval=px_e)

    scoreFile.write("\nThe number of games played was: " + str(2*r) + "\n")
    counte1 = 0
    counte2 = 0
    countTie = 0
    for win in winnerTracker:
        if(win == 'e1'):
            counte1 += 1
        elif(win == 'e2'):
            counte2 += 1
        elif(win == '.'):
            countTie += 1
    scoreFile.write("e1 (Simple Heuristic) won a total of " + str(counte1) + " times, or " + str(counte1/gameCounterTracker*100) + " percent of the time. \n")
    scoreFile.write("e2 (Complex Heuristic) won a total of " + str(counte2) + " times, or " + str(counte2/gameCounterTracker*100) + " percent of the time. \n")
    scoreFile.write("There was a total of " + str(countTie) + " ties.\n")
    scoreFile.write("\nAverage evaluation times: " + str(sum(averageTimeTracker)/len(averageTimeTracker)) + "\n")
    scoreFile.write("Average states evaluated per game: " + str(sum(totalStatesTracker)/len(totalStatesTracker))+ "\n")
    scoreFile.write("Average of average depths: " + str(sum(totalAverageDepthTracker)/len(totalAverageDepthTracker)) + "\n")
    if(g.d1 > g.d2):
        for dd in range (0, g.d1):
            scoreFile.write("Average total states evaluated at depth " + str(dd+1) + ": " + str(totalDepthsTracker[dd]/(2*r)) + "\n")
    else:
        for dd in range (0, g.d2):
            scoreFile.write("Average Total states evaluated at depth " + str(dd+1) + ": " + str(totalDepthsTracker[dd]/(2*r)) + "\n")
    scoreFile.write("Average ARD per game: " + str(sum(totalARDTracker)/len(totalARDTracker)) + "\n")
    scoreFile.write("The average number of moves per game: " + str(sum(totalMovesTracker)/len(totalMovesTracker)) + "\n")

    scoreFile.close

def main():
    game_config = "config.ini"
    #ax, ao,  px, po, px_e, po_e, g = GameBuilder.build_game(game_config)
    #if (g != None):
     #   for _ in (0, 5):
      #      g.play(px_algo = ax, po_algo = ao , player_x=px, player_o=po, px_eval=px_e, po_eval=po_e)
    playrtimes(1, game_config)

if __name__ == "__main__":
    main()