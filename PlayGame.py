# Play game
import torch
import numpy as np
from Environment import ConnectFour3DEnv
from AI_Training import DQN



# Last inn den trente modellen
env = ConnectFour3DEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(env.num_observations, env.num_actions).to(device)
policy_net.load_state_dict(torch.load("dqn_model.pth"))
policy_net.eval()

# Funksjon for å gjøre trekk med modellen
def model_move(state):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        action = policy_net(state_tensor).argmax(dim=1).item()
    return action

# Funksjon for å spille et spill mot modellen
def play_game():
    env = ConnectFour3DEnv()  # Opprett et nytt spillmiljø
    state = env.reset()  # Start et nytt spill
    done = False
    while not done:
        # Spilleren gjør sitt trekk (menneskelig input)

        # Spillerens trekk (input som koordinater)
        valid_move = False
        while not valid_move:
            try:
                # Spilleren gir inn sine trekk som koordinater
                x, y = map(int, input("Enter your move (x y): ").split())
                # Validere trekket og oppdatere brettet
                if env.is_valid_action((x, y)):
                    valid_move = True
                    next_state, reward, done = env.step((x, y))
                else:
                    print("Invalid move, try again.")
            except ValueError:
                print("Invalid input, please enter two integers separated by a space.")

        # Sjekk om spillet er over etter spillerens trekk
        if done:
            print("Player 1 wins!" if env.current_player == 1 else "Player 2 wins!")
            break

        # Modellens trekk
        action = model_move(state)  # Modellens trekk (som en indeks)

        while True:
            action = model_move(state)  # Modellens trekk (som en indeks)
            if env.is_valid_action(env.action_space[action]):

                print(f"AI's move (as index): {action}")
                x, y = env.action_space[action]  # Konverter action til koordinater (x, y)
                print(f"AI's move (as coordinates): ({x}, {y})")

                # Utfør modellens trekk
                state, reward, done = env.step((x, y))
                break

        

        if done and reward < 0:
            print("AI did an invalid move!")
            break

        if done:
            print("Player 1 wins!" if env.current_player == 1 else "Player 2 wins!")
            break
play_game()