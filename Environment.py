import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

TRAINING = False

class ConnectFour3DEnv: # Game environment class
    def __init__(self):
        # init empty grid
        self.grid = np.zeros((4, 4, 4), dtype=int)
        self.current_player = 1  # 1 represents Player 1, -1 represents Player 2
        self.last_placed_piece = None
        self.num_actions = 4 * 4
        self.num_observations = 4 * 4 * 4
        self.action_space = [(x, y) for x in range(4) for y in range(4)] # Action space of all possible combinations of actions
        self.directions = [ # 13 directions in total where four in a row is possible
            (1, 0, 0),  
            (0, 1, 0),  
            (0, 0, 1),  

            (1, 1, 0),  
            (1, 0, 1),  
            (0, 1, 1), 
            (1, -1, 0), 
            (1, 0, -1),
            (0, 1, -1), 
            
            (1, 1, 1),  
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
        ]
        
    def reset(self):
        """Resets the game state."""
        self.grid = np.zeros((4, 4, 4), dtype=int)
        self.current_player = 1
        return self.grid

    def is_valid_action(self, action):
        """Checks if an action (x, y) is valid."""
        # Get action, x, y grid of possible moves
        x, y = action
        return 0 <= x < 4 and 0 <= y < 4 and self.grid[0, x, y] == 0 # z = 0 is the heighest layer, z = 3 is the lowest. If 0, the cell is empty and return true

    def apply_action(self, action):
        """Applies the action (x, y) for the current player. This also calculates the reward (not ideal to have it baked in here)."""
        x, y = action
        reward = 0
        # Place the piece on the lowest available z in the column (x, y). Cannot place pieces in thin air
        for z in range(3, -1, -1):  # iterates from 3 to 0, from the bottom of the board to the top
            if self.grid[z, x, y] == 0:  # first available / empty cell
                # Place the piece by setting it to 1. The current player sees 1 as theirs, and -1 as the opponent's
                self.grid[z, x, y] = 1 

                # Update last placed piece (used to check four in a row)
                self.last_placed_piece = (z, x, y)
                

    def calculate_reward(self):
        """Calculates the reward for the current player."""
        # Calculate reward for adjacent pieces
        reward = 0
        for dx, dy, dz in self.directions:
            count_positive = self._count_in_direction_last_placed_piece(dx, dy, dz)
            count_negative = self._count_in_direction_last_placed_piece(-dx, -dy, -dz)
            
            # Total aligned pieces in this direction (include the placed piece)
            total_count = count_positive + count_negative + 1
            
            # Reward scaling: encourage longer lines
            if total_count == 2:
                reward += 0.5  # Small reward for 2 in a row
            elif total_count == 3:
                reward += 1  # Larger reward for 3 in a row
            elif total_count >= 4:
                reward += 10.0  # Winning move. Game is over

        return reward
    
    def sample_action(self):
        """Returns a random valid action."""
        valid_actions = []
        for x in range(4):
            for y in range(4):
                if self.is_valid_action((x, y)):
                    valid_actions.append((x, y))  # Legg til alle gyldige handlinger
        return random.choice(valid_actions) if valid_actions else None

    def check_win(self):
        """Helper to check for four in a row along any axis or diagonal."""
        for dx, dy, dz in self.directions:
            count = 1  # include the last placed piece

            # Check in positive direction
            count += self._count_in_direction_last_placed_piece(dx, dy, dz)

            # Check in negative direction
            count += self._count_in_direction_last_placed_piece(-dx, -dy, -dz)

            # Return true if four in a row
            if count >= 4:
                return True

        return False

    def _count_in_direction_last_placed_piece(self, dx, dy, dz):
        """Counts all pieces in one direction."""
        z, x, y = self.last_placed_piece

        nx = x + dx
        ny = y + dy
        nz = z + dz

        count = 0
        while 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
            if self.grid[nz, nx, ny] == 1:
                count += 1
                # check next piece
                nx += dx
                ny += dy
                nz += dz
            else:
                break # no need to check further
        return count


    def step(self, action):
        """Performs a step in the environment. Players are switched."""

        # print(f"Before action by player {self.current_player}:\n{self.grid}")
        # print(f"Action: {action}")

        if not self.is_valid_action(action): #TODO: Remove? Validation should be already done
            # invalid action. 
            Exception(f"Invalid action: {action}")

            reward = -1
            done = True
            return self.grid, reward, done
        
        # Apply action
        self.apply_action(action)

        # Calculate reward
        reward = self.calculate_reward()

        # Check for win or draw
        if self.check_win():
            print(f"Player {self.current_player} wins!")
            done = True
        elif np.all(self.grid != 0): # check if the grid is full without a win
            done = True
        else:
            done = False

        if done and reward != 1:
        # If the current player loses (reward is 0), give -1 for the opponent (the player who did not win)
            reward = -10

        # print(f"After action by player {self.current_player}:\n{self.grid}")

        # Switch player if not done
        if not done:
            self.switch_player()
        if not TRAINING:
            print(f"After switching to player {self.current_player}:\n{self.grid}")
        return self.grid, reward, done

    
    
    def switch_player(self):
        """Switches player and inverts perspective."""
        self.current_player = 3 - self.current_player  # Bytter mellom 1 og 2
        self.grid *= -1  # Inverterer perspektivet
