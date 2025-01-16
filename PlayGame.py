# Play game
import torch
import numpy as np
from Environment import ConnectFour3DEnv
from AI_Training import DQN

# init model
env = ConnectFour3DEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(env.num_observations, env.num_actions).to(device)
policy_net.load_state_dict(torch.load("dqn_model.pth"))
policy_net.eval()

def model_move(state, valid_actions):
    """Predicts best move using the model. The move is a valid action."""
    state = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1) # convert state to tensor
    q_values = policy_net(state)[0, :] # Get the Q-values for the current state
    # env.action_space holds all 16 combinations of actions as [(x, y) for x in range(4) for y in range(4)]
    available_action_indices = [env.action_space.index(action) for action in valid_actions] # filter out all invalid actions
    valid_q_values = q_values[available_action_indices]
    
    # Chose highst Q-value of the valid actions
    best_action_idx = valid_q_values.argmax().item()
    best_action = valid_actions[best_action_idx]
    return best_action

def play_game():
    """Plays the game against the model."""
    env = ConnectFour3DEnv()  # Init game environment
    state = env.reset()  # Start empty board
    env.current_player = -1 # Start with player -1 (the human)
    done = False
    while not done:
        # Print board
        print(env.grid)

        # Get human input
        valid_move = False
        while not valid_move:
            try:
                x, y = map(int, input("Enter your move (x y): ").split())
                # Validate move
                if env.is_valid_action((x, y)):
                    valid_move = True
                    state, _, done = env.step((x, y), -1) # human make their move as -1
                else:
                    print("Invalid move, try again.")
            except ValueError:
                print("Invalid input, please enter two integers separated by a space.")

        # Check if game is over
        if done:
            print("AI wins!" if env.current_player == 1 else "You Win!")
            break

        # Model does a move
        valid_actions = env.get_valid_actions()
        action = model_move(state, valid_actions)
        x, y = action
        print(f"AI's move: ({x}, {y})")
        state, _, done = env.step((x, y), 1)

        if done:
            print("Player 1 wins!" if env.current_player == 1 else "Player 2 wins!")
            break
        
play_game()