import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple
from itertools import count
import matplotlib.pyplot as plt
from Environment import ConnectFour3DEnv
import os

# --- Hyperparameters ---
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 2000
TARGET_UPDATE = 10
NUM_EPISODES = 200 * 1000   # Total number of episodes (run up until this number)
LR = 1e-4                   # Learning rate
CHECKPOINT_INTERVAL = 500

# --- DQN Model ---
class DQN(nn.Module):
    """DQN network with two hidden layers."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),  
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, output_dim) 
        )

    def forward(self, x):
        return self.fc(x)



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) #TODO: do i even use this?

# --- Replay Memory ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def push(self, *args):
        """Save a transition to memory."""
        if len(self.memory) < self.memory.maxlen:
            self.memory.append(None)
        self.memory[self.position] = tuple(args)
        self.position = (self.position + 1) % self.memory.maxlen

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = ConnectFour3DEnv()
policy_net = DQN(env.num_observations, env.num_actions).to(device)
target_net = DQN(env.num_observations, env.num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
criterion = nn.SmoothL1Loss()
memory = ReplayMemory(10000)


# --- Optimize Model ---
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    # Sample a batch from memory
    batch = memory.sample(BATCH_SIZE)
    
    # Unpack the batch
    states, actions, next_states, rewards = zip(*batch)

    # Convert to torch tensors and move them to the same device as the model
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)  # Correct dtype for actions
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)

    # Flatten states and next_states to match the expected input size
    states = states.view(states.size(0), -1)  # Flatten (batch_size, 4, 4, 4) to (batch_size, 64)
    next_states = next_states.view(next_states.size(0), -1)  # Same for next_states

    # Calculate Q values for the current state
    q_values = policy_net(states)  # Shape: (batch_size, 16)

    # Get the indices of the selected actions using argmax (since actions are one-hot encoded)
    action_indices = actions.argmax(1)  # Get the indices of the actions (shape: (batch_size,))
    
    # Gather Q values based on the selected actions
    selected_q_values = q_values.gather(1, action_indices.unsqueeze(1))  # Shape: (batch_size, 1)

    # Calculate target Q values
    next_q_values = policy_net(next_states)  # Shape: (batch_size, 16)
    next_q_values = next_q_values.max(1)[0]  # Max Q value for the next state
    target_q_values = rewards + (GAMMA * next_q_values)  # Bellman equation

    # Loss function (e.g., MSE)
    loss = nn.MSELoss()(selected_q_values.squeeze(), target_q_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()





# --- Checkpointing Functions ---
def save_checkpoint(epoch, model, optimizer, steps_done, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps_done': steps_done
    }, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    steps_done = checkpoint['steps_done']
    return epoch, steps_done

# --- Main training loop  ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = ConnectFour3DEnv()
env.is_training = True

policy_net = DQN(env.num_observations, env.num_actions).to(device)
target_net = DQN(env.num_observations, env.num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()
memory = ReplayMemory(10000)


# Select action with epsilon-greedy strategy
def select_action(state, available_actions, steps_done):
    state = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1) # convert state to tensor
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            # select action with policy_net. Chose the action with the highest Q-value that is a valid move
            q_values = policy_net(state)[0, :] # Get the Q-values for the current state
            # env.action_space holds all 16 combinations of actions as [(x, y) for x in range(4) for y in range(4)]
            available_action_indices = [env.action_space.index(action) for action in available_actions] # filter out all invalid actions
            valid_q_values = q_values[available_action_indices]
            
            # Chose highst Q-value of the valid actions
            best_action_idx = valid_q_values.argmax().item()
            best_action = available_actions[best_action_idx]

            return best_action
    else:
        # random action for exploration
        return env.sample_action()

# Start training
if __name__ == '__main__':

    start_episode = 0
    steps_done = 0
    
    # Check if checkpoint exists and load it
    if os.path.exists('checkpoint.pth'):
        start_episode, steps_done = load_checkpoint(policy_net, optimizer)

    # Run until specified number of episodes
    for episode in range(start_episode, NUM_EPISODES + 1):
        # Start with a empty board
        state = env.reset()
        new_state = state
        
        # Make it random who goes first
        env.current_player = random.choice([-1, 1])
        
        for t in count(): # Loop for single game

            if env.current_player == 1: # Select action for player 1
                
                available_actions = env.get_valid_actions() # list of valid actions as an (x y) tuple
                action_p1 = select_action(state, available_actions, steps_done)
                x1, y1 = action_p1

                new_state, reward_p1, done = env.step((x1, y1), 1)  # Player 1 makes their move

                # Store player 1's transition in memory
                action_p1_one_hot = env.action_to_one_hot(action_p1)
                memory.push(state, action_p1_one_hot, new_state, reward_p1)

                if done:
                    break  # End the episode if done
            
            else: # Select action for player 2. Do not train on player 2's actions, only player 1 if loosing
                
                if random.random() < 0.1: # 10% of the time, player 2 makes a random move
                    action_p2 = env.sample_action()
                else:
                    # invert grid for player 2 so the ai can make a move as their "own" perspective
                    new_state *= -1
                    available_actions = env.get_valid_actions() # list of valid actions as an (x y) tuple
                    action_p2 = select_action(new_state, available_actions, steps_done)
                    new_state *= -1 # invert back again

                x2, y2 = action_p2
                new_state, reward_p2, done = env.step((x2, y2), -1)  # Player 2 makes their move as -1

                if done and reward_p2 > 0: # Condition for p1 to loose
                    loss_punishment = -50
                    memory.push(state, action_p1_one_hot, new_state, loss_punishment)
                    
                # Update the state for each player
                state = new_state if not done else None

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                if done:
                    break  # End the episode if done

        # Update the target network every TARGET_UPDATE episodes
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save checkpoint every CHECKPOINT_INTERVAL episodes
        if episode % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(episode, policy_net, optimizer, steps_done)

        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode} completed")

    print("Training complete")

    # Save final model
    torch.save(policy_net.state_dict(), "dqn_model.pth")
    print("Model saved as 'dqn_model.pth'")