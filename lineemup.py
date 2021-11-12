# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time
import numpy as np
import traceback
import sys


class Game:
    MINIMAX = 0
    ALPHABETA = 1
    HUMAN = 2
    AI = 3

    def __init__(self, n, b, barray, s, d1, d2, ai_timeout, recommend=True):
        self.n = n
        self.b = b
        self.barray = barray
        self.s = s
        self.d1 = d1
        self.d2 = d2
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

        self.placesleft = (self.n * self.n) - self.b

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
            while (st_x > 0 and st_y < self.n):
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
        if (self.placesleft > 0):
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

    def minimax(self, depth, start_time, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:
        value = np.inf
        if max:
            value = -np.inf
        x = None
        y = None
        result = self.is_end()
        elapsed_t = time.time() - start_time

        if depth == 0 or result == 'X' or elapsed_t >= self.ai_timeout:
            return (-1, x, y)
        elif depth == 0 or result == 'O' or elapsed_t >= self.ai_timeout:
            return (1, x, y)
        elif depth == 0 or result == '.' or elapsed_t >= self.ai_timeout:
            return (0, x, y)

        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.current_state[i][j] == '.':
                    self.placesleft = self.placesleft - 1
                    
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.minimax(depth-1, start_time, max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.minimax(depth-1, start_time, max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j

                    self.placesleft = self.placesleft + 1
                    self.current_state[i][j] = '.'
        return (value, x, y)

    def alphabeta(self, depth, start_time, alpha=-np.inf, beta=np.inf, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:
        value = np.inf
        if max:
            value = -np.inf
        x = None
        y = None
        result = self.is_end()
        elapsed_t = time.time() - start_time           
        if depth == 0 or result != None or elapsed_t >= self.ai_timeout:
            return (self.simple_heuristic(), x, y)
        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.current_state[i][j] == '.':
                    self.placesleft = self.placesleft - 1

                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.alphabeta(depth-1, start_time, alpha, beta, max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.alphabeta(depth-1, start_time, alpha, beta, max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j

                    self.placesleft = self.placesleft + 1
                    self.current_state[i][j] = '.'

                    if max:
                        if value >= beta:
                            return (value, x, y)
                        if value > alpha:
                            alpha = value
                    else:
                        if value <= alpha:
                            return (value, x, y)
                        if value < beta:
                            beta = value

        return (value, x, y)

    # Simple heuristic
    # The idea is to check rows-columns-diagonals
    # This will add 10 to the score if we see our piece in a row/column/diagonal
    # and subtract 10 to the score if we see an opponents piece
    def simple_heuristic(self, testing=False):
        if (testing == True):
            #self.current_state[0][0] = 'X'
            #self.current_state[1][0] = 'X'
            #self.current_state[4][0] = 'X'
            #self.current_state[3][1] = 'X'
            #self.current_state[2][2] = 'X'
            #self.current_state[4][0] = 'X'
            #self.current_state[3][1] = 'X'
            #self.current_state[2][2] = 'X'
            #self.current_state[1][3] = 'X'
            #self.current_state[3][3] = 'O'
            #self.current_state[4][4] = 'O'
            #self.current_state[1][0] = 'X'
            #self.current_state[0][3] = 'X'
            #self.current_state[0][1] = 'X'
            #self.current_state[2][0] = 'O'
            #self.current_state[3][0] = 'X'
            #self.current_state[4][0] = 'O'
            #self.current_state[2][0] = 'O'
            #self.current_state[0][2] = 'O'
            #self.current_state[0][3] = 'O'
            self.draw_board()

        score = 0

        # Rows
        for y in range(self.n):
            s = self.simple_heuristic_default_values()
            for x in range(self.n):
                s = self.simple_heuristic_evaluator(x, y, s)
            score = score + s

        # Columns
        for x in range(self.n):
            s = self.simple_heuristic_default_values()
            for y in range(self.n):
                s = self.simple_heuristic_evaluator(x, y, s)
            score = score + s

        # Diagonals (left to right)
        for d in range(len(self.diagonal_starting_positions)):
            st_x = self.diagonal_starting_positions[d][0]
            st_y = self.diagonal_starting_positions[d][1]
            s = self.simple_heuristic_default_values()
            while (st_x < self.n and st_y < self.n):
                s = self.simple_heuristic_evaluator(st_x, st_y, s)
                st_x = st_x + 1
                st_y = st_y + 1
            score = score + s

        # Diagonals (right to left)
        for d in range(len(self.other_diagonal_starting_positions)):
            st_x = self.diagonal_starting_positions[d][0]
            st_y = self.diagonal_starting_positions[d][1]
            s = self.simple_heuristic_default_values()
            while (st_x > 0 and st_y < self.n):
                s = self.simple_heuristic_evaluator(st_x, st_y, s)
                st_x = st_x - 1
                st_y = st_y + 1
            score = score + s

        if (testing):
            print(score)

        return score

    def simple_heuristic_default_values(self):
        # [0] Current score for the current row/col/diagonal
        return 0

    def simple_heuristic_evaluator(self, x, y, s):
        if (self.current_state[x][y] == 'O'):
            s = s + 10
        elif (self.current_state[x][y] == 'X'):
            s = s - 10

        return s

    # Complex heuristic
    # The idea is to check all rows/columns/diagonals
    # If we have a 'streak' going (a row/column/diagonal without an opposing piece)
    # the score will be exponentially increased
    # If a move will be blocked by opponent piece, a wall, or a block, the streak is over and no score increase (TODO)
    def complex_heuristic(self, testing=False):
        if (testing == True):
            #self.current_state[0][0] = 'X'
            #self.current_state[1][0] = 'O'
            #self.current_state[4][0] = 'X'
            #self.current_state[3][1] = 'X'
            #self.current_state[2][2] = 'X'
            #self.current_state[4][0] = 'X'
            #self.current_state[3][1] = 'X'
            #self.current_state[2][2] = 'X'
            #self.current_state[1][3] = 'X'
            #self.current_state[3][3] = 'O'
            #self.current_state[4][4] = 'O'
            #self.current_state[1][0] = 'X'
            #self.current_state[0][3] = 'X'
            #self.current_state[0][1] = 'X'
            #self.current_state[2][0] = 'O'
            #self.current_state[3][0] = 'X'
            #self.current_state[4][0] = 'O'
            #self.current_state[2][0] = 'O'
            #self.current_state[0][2] = 'O'
            #self.current_state[0][3] = 'O'
            self.draw_board()

        score = 0

        # Rows
        for y in range(self.n):
            s, m, cs, prevState = self.complex_heuristic_default_values()
            for x in range(self.n):
                s, m, cs, prevState = self.complex_heuristic_evaluator(x, y, s, m, cs, prevState)
            score = score + s

        # Columns
        for x in range(self.n):
            s, m, cs, prevState = self.complex_heuristic_default_values()
            for y in range(self.n):
                s, m, cs, prevState = self.complex_heuristic_evaluator(x, y, s, m, cs, prevState)
            score = score + s

        # Diagonal (left to right)
        s, m, cs, prevState = self.complex_heuristic_default_values()
        for x in range(self.n):
            s, m, cs, prevState = self.complex_heuristic_evaluator(x, x, s, m, cs, prevState)
            score = score + s

        # Diagonal (right to left)
        s, m, cs, prevState = self.complex_heuristic_default_values()
        for x in range(self.n):
            s, m, cs, prevState = self.complex_heuristic_evaluator(self.n - x - 1, x, s, m, cs, prevState)
            score = score + s

        if (testing):
            print(score)

        return score

    def complex_heuristic_default_values(self):
        # [0] Current score for the current row/col/diagonal
        # [1] Score multiplier
        # [2] Consecuive streak
        # [3] Previous state
        return (0, 2, 0, "~")

    def complex_heuristic_evaluator(self, x, y, s, m, cs, prevState):
        if (self.current_state[x][y] == 'O'):
            if (prevState == 'O'):
                m = m << 1
            elif (prevState == 'X'):
                s = s + (m >> cs) # a move which blocks the opposing color streak is a good move for us
                m = 2
                cs = 0

            s = s + m
            cs = cs + 1
            prevState = self.current_state[x][y]
        elif (self.current_state[x][y] == 'X'):
            if (prevState == 'X'):
                m = m << 1
            elif (prevState == 'O'):
                s = s - (m >> cs) # a move which blocks our color streak is a bad move for us (good move for opponent)
                m = 2
                cs = 0

            s = s - m
            cs = cs + 1
            prevState = self.current_state[x][y]

        return (s, m, cs, prevState)

    def play(self, algo=None, player_x=None, player_o=None):
        if algo == None:
            algo = self.ALPHABETA
        if player_x == None:
            player_x = self.HUMAN
        if player_o == None:
            player_o = self.HUMAN

        while True:
            self.draw_board()
            if self.check_end():
                return

            start = time.time()
            if algo == self.MINIMAX:
                if self.player_turn == 'X':
                    (_, x, y) = self.minimax(self.d1, start, max=False)
                else:
                    (_, x, y) = self.minimax(self.d2, start, max=True)
            elif algo == self.ALPHABETA:
                if self.player_turn == 'X':
                    (m, x, y) = self.alphabeta(self.d1, start, max=False)
                else:
                    (m, x, y) = self.alphabeta(self.d2, start, max=True)

            end = time.time()

            if (player_o == self.AI and (end - start) > self.ai_timeout) or (player_x == self.AI and (end - start) > self.ai_timeout):
                if self.player_turn == 'X':
                    self.result = 'O'
                else:
                    self.result = 'X'
                print(F"Player {('X' if self.player_turn == 'X' else 'O')} has taken too long to make a move.")
                print(F"The winner is {('O' if self.player_turn == 'X' else 'X')}!")
                return

            if (self.player_turn == 'X' and player_x == self.HUMAN) or (self.player_turn == 'O' and player_o == self.HUMAN):
                if self.recommend:
                    print(F'Evaluation time: {round(end - start, 7)}s')
                    print(F'Recommended move: {self.alphabet[x]} {y}')

                (x, y) = self.input_move()
            if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
                print(F'Evaluation time: {round(end - start, 7)}s')
                print(F'Player {self.player_turn} under AI control plays: {self.alphabet[x]} {y}')

            self.current_state[x][y] = self.player_turn
            self.placesleft = self.placesleft - 1

            self.switch_player()

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
                        temp = iter(line.split('=')[1].split())
                        block_array = [(int(ele.split(",")[0]), int(ele.split(",")[1])) for ele in temp]
                    elif (line.split('=')[0] == "winLength"):
                        win_length = int(line.split('=')[1])
                    elif (line.split('=')[0] == "maxDepthD1"):
                        max_depthD1 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "maxDepthD2"):
                        max_depthD2 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "aiTimeout"):
                        ai_timeout = int(line.split('=')[1])
                    elif (line.split('=')[0] == "alphabeta"):
                        alphabeta = int(line.split('=')[1])
                    elif (line.split('=')[0] == "p1"):
                        p1 = int(line.split('=')[1])
                    elif (line.split('=')[0] == "p2"):
                        p2 = int(line.split('=')[1])
                
                config.close()                   
        except Exception:
            print("ERROR: Could not open file", config_path) 
            print(traceback.format_exc())
            return (None, None, None, None)

        # Size of board
        if (board_size < 3 or board_size > 10):
            print("ERROR: Invalid game configuration. Board size must be between 3 and 10 inclusive")    
            return (None, None, None, None)

        # Number of blocks
        if (len(block_array) != block_count):
            print("ERROR: Invalid game configuration. Number of blocks does not match size of block array in config file")    
            return (None, None, None, None)

        # Number of blocks
        if (block_count < 0 or block_count > 2 * board_size):
            print(F"ERROR: Invalid game configuration. Number of blocks placed must be between 0 and {2 * board_size} inclusive")    
            return (None, None, None, None)

        # Validation for the blocks placed
        for x in range(len(block_array)):
            block_placed = block_array[x]
            if (block_placed[0] >= board_size or block_placed[1] >= board_size):
                print(F"ERROR: Invalid game configuration. Block {block_placed} not placed inside board")    
                return (None, None, None, None)

        # Winning line up size
        if (win_length < 3 or win_length > board_size):
            print(F"ERROR: Invalid game configuration. Winning line up size must be between 3 and {board_size} inclusive")    
            return (None, None, None, None)

        return (alphabeta, p1, p2, Game(board_size, block_count, block_array, win_length, max_depthD1, max_depthD2, ai_timeout, recommend=True))

def main():
    game_config = "config.ini"    
    algorithm, px, py, g = GameBuilder.build_game(game_config)
    if (g != None):
        g.play(algo=algorithm, player_x=px, player_o=py)

if __name__ == "__main__":
    main()
