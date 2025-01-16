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

def model_move(state):
    """Predicts a move using the model."""
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        action = policy_net(state_tensor).argmax(dim=1).item()
    return action

def play_game():
    """Plays a game against the model."""
    env = ConnectFour3DEnv()  # Init environment
    state = env.reset()  # Starts a new game with an empty board
    done = False
    while not done:

        # Player input must be valid
        valid_move = False
        while not valid_move:
            try:
                # Get player input
                x, y = map(int, input("Enter your move (x y): ").split())
                # Validate move
                if env.is_valid_action((x, y)):
                    valid_move = True
                    _, _, done = env.step((x, y)) # make the move
                else:
                    print("Invalid move, try again.")
            except ValueError:
                print("Invalid input, please enter two integers separated by a space.")

        # Check if game is over
        if done:
            print("Player 1 wins!" if env.current_player == 1 else "Player 2 wins!")
            break

        # Model predicts a move / action
        action = model_move(state)
        print(f"AI's move (as index): {action}")
        x, y = env.action_space[action]  # Convert from index to coordinates (x, y)
        print(f"AI's move (as coordinates): ({x}, {y})")

        # Make the move in the environment
        state, _, done = env.step((x, y))

        if done:
            print("Player 1 wins!" if env.current_player == 1 else "Player 2 wins!")
            break

        
play_game()