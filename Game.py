import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class ConnectFour3DEnv: # Game environment class
    def __init__(self):
        # init empty grid
        self.grid = np.zeros((4, 4, 4), dtype=int)
        self.current_player = 1  # 1 for Player 1, 2 for Player 2
        self.last_placed_brick = None

    def reset(self):
        """Resets the game state."""
        self.grid = np.zeros((4, 4, 4), dtype=int)
        self.current_player = 1
        return self.grid

    def is_valid_action(self, action):
        """Checks if an action (x, y) is valid."""
        # Get action, x, y grid of possible moves
        x, y = action
        return 0 <= x < 4 and 0 <= y < 4 and self.grid[0, x, y] == 0 # z = 0 is the heighest layer, z = 3 is the lowest

    def apply_action(self, action):
        """Applies the action (x, y) for the current player."""
        x, y = action
        # Find the lowest available z in the column (x, y)
        for z in range(3, -1, -1): # iterates from 3 to 0, bottom of the board to top
            if self.grid[z, x, y] == 0: # if the cell is empty
                self.grid[z, x, y] = 1 # the current ai sees 1 as theirs and -1 as the opponents
                # remember the coordinates of the last placed brick
                self.last_placed_brick = (z, x, y)
                break
    

    def check_win(self):
        """Helper to check for four in a row along any axis or diagonal."""
        directions = [ # 13 directions in total
            # (1, 0, 0),  
            # (0, 1, 0),  
            (0, 0, 1),  

            # (1, 1, 0),  
            # (1, 0, 1),  
            # (0, 1, 1), 
            # (1, -1, 0), 
            # (1, 0, -1),
            # (0, 1, -1), 
            
            # (1, 1, 1),  
            # (1, -1, -1),
            # (-1, 1, -1),
            # (-1, -1, 1),
        ]
        
        for dx, dy, dz in directions:
            count = 1  # include the last placed brick

            # Check in positive direction
            count += self._count_in_direction(dx, dy, dz)

            # Check in negative direction
            count += self._count_in_direction(-dx, -dy, -dz)

            # Return true if four in a row
            if count >= 4:
                return True

        return False

    def _count_in_direction(self, dx, dy, dz):
        """Counts all bricks in one direction."""
        z, x, y = self.last_placed_brick

        nx = x + dx
        ny = y + dy
        nz = z + dz

        count = 0
        while 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
            if self.grid[nz, nx, ny] == 1:
                count += 1
                # check next brick
                nx += dx
                ny += dy
                nz += dz

                print("count", count)
            else:
                break # no need to check further
        
        return count


    # TODO: look over this function
    def step(self, action):
        """Performs a step in the environment."""
        if not self.is_valid_action(action):
            raise ValueError("Invalid action")
            # TODO: this should be considered as a loss, not an error

        self.apply_action(action)

        # Check for win or draw
        if self.check_win():
            reward = 1  # Win for current player
            done = True
        elif np.all(self.grid != 0):
            reward = 0  # Draw
            done = True
        else:
            reward = 0
            done = False

        # Switch player
        self.switch_player()
        return self.grid, reward, done

    
    def switch_player(self):
        # switch player as seen from the outside
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
        
        # Invert perspective. The AI always sees 1 as theirs and -1 as the opponents
        self.grid *= -1





game = ConnectFour3DEnv()
game.apply_action([3, 3])
game.apply_action([3, 3])
game.apply_action([3, 3])
game.apply_action([3, 3])



print(game.check_win())
print(game.grid)
