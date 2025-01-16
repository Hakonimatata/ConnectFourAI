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

# Funksjon for å gjøre trekk med modellen
def model_move(state, valid_actions):
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
    env = ConnectFour3DEnv()  # Opprett et nytt spillmiljø
    state = env.reset()  # Start et nytt spill
    env.current_player = -1 # Start med spiller 2, mennesket
    done = False
    while not done:
        # Spilleren gjør sitt trekk (menneskelig input)
        print(env.grid)
        # Spillerens trekk (input som koordinater)
        valid_move = False
        while not valid_move:
            try:
                # Get player input
                x, y = map(int, input("Enter your move (x y): ").split())
                # Validate move
                if env.is_valid_action((x, y)):
                    valid_move = True
                    state, reward, done = env.step((x, y), -1)
                else:
                    print("Invalid move, try again.")
            except ValueError:
                print("Invalid input, please enter two integers separated by a space.")

        # Check if game is over
        if done:
            print("AI wins!" if env.current_player == 1 else "You Win!")
            break

        # Modellens trekk
        valid_actions = env.get_valid_actions() # list of valid actions as an (x y) tuple
        action = model_move(state, valid_actions)  # Modellens trekk (som en indeks)
        x, y = action  # Konverter action til koordinater (x, y)
        print(f"AI's move: ({x}, {y})")

        # Utfør modellens trekk
        state, reward, done = env.step((x, y), 1)

        

        if done and reward < 0:
            print("AI did an invalid move!")
            break

        if done:
            print("Player 1 wins!" if env.current_player == 1 else "Player 2 wins!")
            break

        
play_game()