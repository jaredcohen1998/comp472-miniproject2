# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time
import numpy as np


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
        self.initialize_game()
        self.recommend = recommend        

    def initialize_game(self):
        self.current_state = np.full((self.n, self.n), '.')   
        print(self.current_state)                    
        for x in self.b:           
            self.current_state[x[0]][x[1]] = '~'

        # Player X always plays first
        self.player_turn = 'X'

    def draw_board(self):
        print()
        for y in range(0, self.n):
            for x in range(0, self.n):
                print(F'{self.current_state[x][y]}', end="")
            print()
        print()

    def is_valid(self, px, py):
        if px < 0 or px > self.n-1 or py < 0 or py > self.n-1:
            return False
        elif self.current_state[px][py] != '.':
            return False
        else:
            return True

    def is_end(self):
        # Vertical win
        for i in range(0, self.n):
            l = 0
            di = -1
            dj = -1            
            for j in range(0, self.n):
                if (l == self.s):
                    return self.current_state[dj][di]
                elif (l == 0 and (self.current_state[j][i] != '.' and self.current_state[j][i] != '~')):
                    di = i
                    dj = j
                    l += 1
                elif ((di > -1 and dj > -1) and (self.current_state[dj][di] == self.current_state[j][i])):
                    l += 1
                else:
                    l = 0                    
        # Horizontal win
        for i in range(0, self.n):
            l = 0
            di = -1
            dj = -1            
            for j in range(0, self.n):
                if (l == self.s):
                    return self.current_state[di][dj]
                elif (l == 0 and (self.current_state[i][j] != '.' and self.current_state[i][j] != '~')):
                    di = i
                    dj = j
                    l += 1
                elif ((di > -1 and dj > -1) and (self.current_state[di][dj] == self.current_state[i][j])):
                    l += 1
                else:
                    l = 0    
        # Main diagonal win
        l = 0
        di = -1
        for i in range(0, self.n):
            if (l == self.s):
                return self.current_state[di][di]
            elif (l == 0 and (self.current_state[i][i] != '.' and self.current_state[i][i] != '~')):
                di = i
                l += 1
            elif (di > -1 and self.current_state[di][di] == self.current_state[i][i]):
                l += 1
            else:
                l = 0        
        # Second diagonal win
        l = 0
        di = -1        
        for i in range(0, self.n):
            if (l == self.s):
                return self.current_state[self.n-(di+1)][di]
            elif (l == 0 and (self.current_state[self.n-(i+1)][i] != '.' and self.current_state[self.n-(i+1)][i] != '~')):
                di = i
                l += 1
            elif (di > -1 and self.current_state[self.n-(di+1)][di] == self.current_state[self.n-(i+1)][i]):
                l += 1
            else:
                l = 0   
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
        while True:
            print(F'Player {self.player_turn}, enter your move:')
            px = int(input('enter the x coordinate: '))
            py = int(input('enter the y coordinate: '))
            if self.is_valid(px, py):
                return (px, py)
            else:
                print('The move is not valid! Try again.')

    def switch_player(self):
        if self.player_turn == 'X':
            self.player_turn = 'O'
        elif self.player_turn == 'O':
            self.player_turn = 'X'
        return self.player_turn

    def minimax(self, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:
        value = 2
        if max:
            value = -2
        x = None
        y = None
        result = self.is_end()
        if result == 'X':
            return (-1, x, y)
        elif result == 'O':
            return (1, x, y)
        elif result == '.':
            return (0, x, y)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.minimax(max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.minimax(max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = '.'
        return (value, x, y)

    def alphabeta(self, alpha=-2, beta=2, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:
        value = 2
        if max:
            value = -2
        x = None
        y = None
        result = self.is_end()
        if result == 'X':
            return (-1, x, y)
        elif result == 'O':
            return (1, x, y)
        elif result == '.':
            return (0, x, y)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.alphabeta(alpha, beta, max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.alphabeta(alpha, beta, max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j
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
                    (_, x, y) = self.minimax(max=False)
                else:
                    (_, x, y) = self.minimax(max=True)
            else:  # algo == self.ALPHABETA
                if self.player_turn == 'X':
                    (m, x, y) = self.alphabeta(max=False)
                else:
                    (m, x, y) = self.alphabeta(max=True)
            end = time.time()
            if (self.player_turn == 'X' and player_x == self.HUMAN) or (self.player_turn == 'O' and player_o == self.HUMAN):
                if self.recommend:
                    print(F'Evaluation time: {round(end - start, 7)}s')
                    print(F'Recommended move: x = {x}, y = {y}')
                (x, y) = self.input_move()
            if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
                print(F'Evaluation time: {round(end - start, 7)}s')
                print(
                    F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
            self.current_state[x][y] = self.player_turn
            self.switch_player()

class GameBuilder:                    
    
    def build_game(config_path):        
        try:
            with open(config_path, 'r') as config:
                for line in config:                    
                    if (line.split('=')[0].equals("boardSize")):
                        board_size = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("blockCount")):
                        block_count = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("blockArray")):
                        block_array = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("winLength")):
                        win_length = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("maxDepthD1")):
                        max_depthD1 = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("maxDepthD2")):
                        max_depthD2 = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("aiTimeout")):
                        ai_timeout = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("alphabeta")):
                        alphabeta = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("p1")):
                        p1 = line.split('=')[1]                        
                    elif (line.split('=')[0].equals("p2")):
                        p2 = line.split('=')[1]
                config.close()                   
        except:
            print("ERROR: Could not open file ", config_path)                  
        if (block_count <= 2*board_size and len(block_array) == block_count and win_length >= 3 and win_length <= board_size):
            return(alphabeta, p1, p2, Game(board_size, block_count, block_array, win_length, max_depthD1, max_depthD2, ai_timeout, recommend=True))
        else:
            print("ERROR: Invalid game configuration.")      

def main():
    game_config = 'config.ini'
    algo, player_x, player_y, g = GameBuilder.build_game(game_config)
    if (g != None):
        g.play(algo=Game.ALPHABETA, player_x=Game.AI, player_o=Game.AI)
        g.play(algo=Game.MINIMAX, player_x=Game.AI, player_o=Game.HUMAN)


if __name__ == "__main__":
    main()
