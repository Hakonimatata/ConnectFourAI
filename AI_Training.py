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
EPS_DECAY = 1000
TARGET_UPDATE = 10
NUM_EPISODES = 20 * 1000
LR = 1e-4
CHECKPOINT_INTERVAL = 500


# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Første skjulte lag
            nn.ReLU(),
            nn.Linear(128, 128),  # Andre skjulte lag
            nn.ReLU(),
            nn.Linear(128, 128),  # Tredje skjulte lag (nytt lag)
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Utgangslag
        )

    def forward(self, x):
        return self.fc(x)



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state)  # Stack the states into a single batch tensor
    action_batch = torch.tensor(batch.action, device=device).view(-1, 1)  # Ensure action_batch has shape (batch_size, 1)
    reward_batch = torch.tensor(batch.reward, device=device)
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # Check the shape of the output of policy_net(state_batch)
    q_values = policy_net(state_batch)  # This should have shape (batch_size, num_actions)

    # Squeeze q_values to remove the singleton dimension
    q_values = q_values.squeeze(1)  # Now shape should be (batch_size, num_actions)

    # Gather the state-action values from the policy network
    state_action_values = q_values.gather(1, action_batch)  # action_batch should be of shape (batch_size, 1)

    # Squeeze to remove the extra dimension (it will become of shape (batch_size,))
    state_action_values = state_action_values.squeeze(1)  # Remove the unnecessary second dimension

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Calculate the loss
    loss = criterion(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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

# --- Main training loop. Run thus for training ---
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
def select_action(state, steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.num_actions)]], device=device, dtype=torch.long)

# Start training
if __name__ == '__main__':

    start_episode = 0
    steps_done = 0
    
    # Check if checkpoint exists and load it
    if os.path.exists('checkpoint.pth'):
        start_episode, steps_done = load_checkpoint(policy_net, optimizer)

    # Run until specified number of episodes
    for episode in range(start_episode, NUM_EPISODES):
        state_p1 = torch.tensor(env.reset(), dtype=torch.float32, device=device).view(1, -1)
        state_p2 = torch.tensor(env.reset(), dtype=torch.float32, device=device).view(1, -1)  # Player 2 starts with the same state
        
        for t in count():
            # Select action for player 1
            # available_actions = env.get_available_actions()
            action_p1 = select_action(state_p1, steps_done)
            
            # Convert action to coordinates for player 1
            x1, y1 = env.action_space[action_p1.item()]

            next_state_p1, reward_p1, done = env.step((x1, y1), 1)  # Player 1 makes their move
            next_state_p1 = None if done else torch.tensor(next_state_p1, dtype=torch.float32, device=device).view(1, -1)
            reward_p1 = torch.tensor([reward_p1], device=device)

            # Store player 1's transition in memory
            memory.push(state_p1, action_p1, next_state_p1, reward_p1)

            if done:
                break  # End the episode if done

            # Select action for player 2
            action_p2 = env.sample_action()
            next_state_p2, reward_p2, done = env.step(action_p2, 2)  # Player 2 makes their move
            
            if reward_p2 >= 10: # punish ai for loosing
                memory.push(state_p1, action_p1, next_state_p1, -20)
                
            next_state_p2 = None if done else torch.tensor(next_state_p2, dtype=torch.float32, device=device).view(1, -1)

            # Update the state for each player
            state_p1 = torch.tensor(next_state_p1, dtype=torch.float32, device=device).view(1, -1) if not done else None
            state_p2 = torch.tensor(next_state_p2, dtype=torch.float32, device=device).view(1, -1) if not done else None

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

        print(f"Episode {episode} completed")

    print("Training complete")

    # Save final model
    torch.save(policy_net.state_dict(), "dqn_model.pth")
    print("Model saved as 'dqn_model.pth'")