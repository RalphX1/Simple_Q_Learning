import numpy as np
import time
import random
import keyboard
import matplotlib.pyplot as plt

# Load the learned Q-table
Q_table = np.load('final_Q_table.npy')

# Define the grid size and initialize the agent's position
GRID_SIZE = 5
REWARDS = 0
agent_position = [0, 0]  # [row, column]

# Define the function to get the state index from the agent's position
def get_state_index(agent_position):
    return agent_position[0] * GRID_SIZE + agent_position[1]

# Define the function to choose the best action for a given state
def choose_best_action(state):
    return np.argmax(Q_table[state])

# Define the function to perform the action and update the agent's position
def perform_action(action):
    global agent_position
    global REWARDS

    # Map the action index to a direction
    directions = ['w', 's', 'a', 'd']  # up, down, left, right
    direction = directions[action]

    # Perform the action
    if direction == 'w' and agent_position[0] > 0:  # 'w' for up
        agent_position[0] -= 1
        print("going up")
    elif direction == 's' and agent_position[0] < GRID_SIZE - 1:  # 's' for down
        agent_position[0] += 1
        print("going down")
    elif direction == 'a' and agent_position[1] > 0:  # 'a' for left
        agent_position[1] -= 1
        print("going left")
    elif direction == 'd' and agent_position[1] < GRID_SIZE - 1:  # 'd' for right
        agent_position[1] += 1
        print("gonig right")
    else:
        print("Invalid move")

    # Check if the agent has reached the reward position
    if agent_position == [1,4]:
        REWARDS += 7
        print("YEAH!, I'll received a REWARD!")
        # Transport the agent to a random position within the grid
        agent_position[0] = random.randint(0, GRID_SIZE - 1)
        agent_position[1] = random.randint(0, GRID_SIZE - 1)
        return 7
    else:
        return 0


# Define the function to print the grid world
def print_grid():
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    grid[agent_position[0]][agent_position[1]] = 'A'

    #add position of reward to the grid
    grid[1][4] = 7

    for row in grid:
        print(' '.join(str(item) for item in row))

# Main loop
while True:
    # Print the grid world
    print("To stop agent, press '7' (press it long).")
    print("Current Agent Rewards =", REWARDS)
    print_grid()

    # Get the current state
    state = get_state_index(agent_position)

    # Choose the best action
    action = choose_best_action(state)

    # Perform the action
    perform_action(action)

    # Wait for 2 seconds
    time.sleep(2)

    if keyboard.is_pressed('7'):  # If the '7' key is pressed 
            print('Key 7 clicked! - EXIT')
            break  # Stop the loop

    # Check if the user wants to quit
    # user_input = input("Press 'q' to quit, or any other key to continue: ")
    # if user_input.lower() == 'q':
    #    break


"""
#show learned Q-Table
print(Q_table)

# Create a new figure with a specified size
plt.figure(figsize=(15, 10))  # Adjust the size as needed

# Assuming Q_table is your Q-table
plt.imshow(Q_table, cmap='hot', interpolation='nearest')
plt.colorbar(label='Q-value')
plt.title('Q-table Heatmap')

# Add numerical values to each cell
for i in range(Q_table.shape[0]):
    for j in range(Q_table.shape[1]):
        plt.text(j, i, np.round(Q_table[i, j], 2), ha='center', va='center', color='black', fontsize=8)  # Adjust the font size as needed

plt.show()
"""