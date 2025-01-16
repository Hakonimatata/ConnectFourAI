import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

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
            (1, 1, 0), (1, 0, 1), (0, 1, 1), 
            (1, -1, 0), (1, 0, -1), (0, 1, -1), 
            (1, 1, 1), (1, -1, -1), (-1, 1, -1),
            (-1, -1, 1),
        ]
        self.is_training = False
        
    def reset(self):
        """Resets the game state."""
        self.grid = np.zeros((4, 4, 4), dtype=int)
        self.current_player = 1
        return self.grid
    
    def is_valid_action(self, action):
        """Checks if an action (x, y) is valid."""
        x, y = action

        # Is the action within the grid?
        if not (0 <= x < 4 and 0 <= y < 4):
            return False

        # Is there a free spot for the piece?
        for z in range(3, -1, -1):  # z=3 is the bottom while z=0 is the top
            if self.grid[z, x, y] == 0: # Return true if there is a free spot
                return True  

        return False  # Ingen ledige plasser i kolonnen


    def apply_action(self, action):
        """Applies the action (x, y) for the current player."""
        x, y = action
        # Place the piece on the lowest available z in the column (x, y). Cannot place pieces in thin air
        for z in range(3, -1, -1):  # iterates from 3 to 0, from the bottom of the board to the top
            if self.grid[z, x, y] == 0:  # first available / empty cell
                # Place the piece by setting it to current player
                self.grid[z, x, y] = self.current_player

                # Update last placed piece (used to check four in a row)
                self.last_placed_piece = (z, x, y)
                break # Exit loop after placing the piece
                

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
                reward += 0.1  # Small reward for 2 in a row
            elif total_count == 3:
                reward += 1  # Larger reward for 3 in a row
            elif total_count >= 4:
                reward += 10.0  # Winning move. Game is over


            # Calculate reward for blocking the opponent's 4 in a row
            count_positive = self._count_in_direction_last_placed_piece(dx, dy, dz, count_opponent = True)
            count_negative = self._count_in_direction_last_placed_piece(-dx, -dy, -dz, count_opponent = True)
            total_count = count_positive + count_negative

            if total_count == 3:
                reward += 10  # Reward for blocking

        return reward
    
    def sample_action(self):
        """Returns a random valid action."""
        valid_actions = self.get_valid_actions()
        return random.choice(valid_actions) if valid_actions else None


    def get_valid_actions(self):
        """Returns a list of all valid actions."""
        valid_actions = []
        for x in range(4):
            for y in range(4):
                if self.is_valid_action((x, y)):
                    valid_actions.append((x, y))
        return valid_actions


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

    def _count_in_direction_last_placed_piece(self, dx, dy, dz, count_opponent=False):
        """Counts all pieces in one direction. It can also be used to count the opponent's pieces from the direction of the last placed piece."""
        z, x, y = self.last_placed_piece

        nx = x + dx
        ny = y + dy
        nz = z + dz

        piece_to_count = -self.current_player if count_opponent else self.current_player

        count = 0
        while 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
            if self.grid[nz, nx, ny] == piece_to_count:
                count += 1
                # check next piece
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

        if not self.is_valid_action(action): # This should never happen
            # invalid action. 
            Exception(f"Invalid action: {action}")

        # Apply the action for the given player
        self.apply_action(action)
        
        # Calculate reward
        reward = self.calculate_reward()

        # Check for win or draw after the action is applied
        if self.check_win():
            done = True
        elif np.all(self.grid != 0):  # End if draw
            reward = 0
            done = True
        else:
            done = False
            # Switch player
            self.switch_player()

        return self.grid, reward, done

    
    def switch_player(self):
        """Switches the current player. 1 represents current player, -1 represents opponent."""
        self.current_player *= -1

    def action_to_one_hot(self, action):
        """Converts an action (x, y) to a one-hot vector."""
        action_index = self.action_space.index(action)  # Finds index of chosen action
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action_index] = 1
        return one_hot_action
    
    def one_hot_to_action(self, one_hot_action):
        """Converts a one-hot vector to an action (x, y)."""
        action_index = np.argmax(one_hot_action)
        return self.action_space[action_index]

