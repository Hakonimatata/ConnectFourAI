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
        self.num_actions = 4 * 4
        self.num_observations = 4 * 4 * 4
        self.is_training = False
        # Create the action space of all possible combinations of actions
        self.action_space = [(x, y) for x in range(4) for y in range(4)]
        self.directions = [ # 13 directions in total
            (1, 0, 0), (0, 1, 0), (0, 0, 1),  
            (1, 1, 0), (1, 0, 1), (0, 1, 1), 
            (1, -1, 0), (1, 0, -1), (0, 1, -1), 
            (1, 1, 1), (1, -1, -1), (-1, 1, -1),
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
        return 0 <= x < 4 and 0 <= y < 4 and self.grid[0, x, y] == 0 # z = 0 is the heighest layer, z = 3 is the lowest

    def apply_action(self, action):
        """Applies the action (x, y) for the current player."""
        x, y = action
        reward = 0  # Initialize reward
        # Find the lowest available z in the column (x, y)
        for z in range(3, -1, -1):  # iterates from 3 to 0, bottom of the board to top
            if self.grid[z, x, y] == 0:  # if the cell is empty
                self.grid[z, x, y] = self.current_player  # Place the brick as the current player
                self.last_placed_brick = (z, x, y)
                
                # Calculate reward for adjacent bricks
                for dx, dy, dz in self.directions:
                    count_positive = self._count_in_direction_last_placed_brick(dx, dy, dz)
                    count_negative = self._count_in_direction_last_placed_brick(-dx, -dy, -dz)
                    
                    # Total aligned bricks in this direction (include the placed brick)
                    total_count = count_positive + count_negative + 1
                    
                    # Reward scaling: encourage longer lines
                    if total_count == 2:
                        reward += 0.5  # Small reward for 2 in a row
                    elif total_count == 3:
                        reward += 1  # Larger reward for 3 in a row
                    elif total_count >= 4:
                        reward += 10.0  # Winning move

                break  # Exit loop after placing the brick
        return reward

    
    def sample_action(self):
        """Returns a random valid action."""
        valid_actions = self.get_available_actions()
        return random.choice(valid_actions) if valid_actions else None


    def get_available_actions(self):
        """Returns a list of all valid actions."""
        valid_actions = []
        for x in range(4):
            for y in range(4):
                if self.is_valid_action((x, y)):
                    valid_actions.append((x, y))  # Legg til alle gyldige handlinger
        return valid_actions


    def check_win(self):
        """Helper to check for four in a row along any axis or diagonal."""
        for dx, dy, dz in self.directions:
            count = 1  # include the last placed brick

            # Check in positive direction
            count += self._count_in_direction_last_placed_brick(dx, dy, dz)

            # Check in negative direction
            count += self._count_in_direction_last_placed_brick(-dx, -dy, -dz)

            # Return true if four in a row
            if count >= 4:
                return True

        return False

    def _count_in_direction_last_placed_brick(self, dx, dy, dz):
        """Counts all bricks in one direction."""
        z, x, y = self.last_placed_brick

        nx = x + dx
        ny = y + dy
        nz = z + dz

        count = 0
        while 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
            if self.grid[nz, nx, ny] == self.current_player:
                count += 1
                # check next brick
                nx += dx
                ny += dy
                nz += dz
            else:
                break # no need to check further
        return count


    def step(self, action, player):
        """Performs a step in the environment for a given player."""
        if self.current_player != player:
            raise ValueError(f"It's player {self.current_player}'s turn, not player {player}'s.")

        if not self.is_valid_action(action):
            # Invalid action. Penalize the AI and return
            reward = -20
            done = True
            return self.grid, reward, done

        # Action is valid, apply the action for the given player
        reward = self.apply_action(action)
        
        # Check for win or draw after the action is applied
        if self.check_win():
            reward = 10
            done = True
        elif np.all(self.grid != 0):  # check if the grid is full without a win
            reward = 0  # Draw
            done = True
        else:
            reward = 0
            done = False

        if not done:
            self.switch_player()

        return self.grid, reward, done


    
    def switch_player(self):
        """Switches the current player."""
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

