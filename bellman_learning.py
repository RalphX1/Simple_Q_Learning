import random
import numpy as np


# Create a simple Reinforcement Learning environment
#Create a small field 5x5 
# Initialize the grid size
GRID_SIZE = 5

"""
This creates a list of lists (a 2D list),
where each inner list is a row in the grid.
range(GRID_SIZE) generates a sequence of numbers from 0 up to (but not including) GRID_SIZE,
and for each number in that sequence, 0 is added to the inner list.
This is done GRID_SIZE times to create GRID_SIZE number of rows,
resulting in a GRID_SIZE x GRID_SIZE grid filled with zeros.
"""
# Create the grid
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# define position of reward inside the grid
grid[1][4] = 7

# Now let's creathe the agent (Agent R)
AGENT_R = 8

# Add rewards counter 
REWARDS = 0

# Define the epsilon value (for agent to explore environment with randdom actions) 
epsilon = 0.3

# Initial position of the agent
agent_position = [0, 0]  # [row, column]

# Function to update the grid with the agent's position
def update_grid():
    global grid  # Declare grid as global
    # First, reset the grid to all zeros
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    #add postition of reqard to the grid
    grid[1][4] = 7
    # Then, set the agent's position to AGENT_R
    grid[agent_position[0]][agent_position[1]] = AGENT_R


# Function to move the agent
def move_agent(direction):
    if direction == 'w' and agent_position[0] > 0:  # 'w' for up
        agent_position[0] -= 1
    elif direction == 's' and agent_position[0] < GRID_SIZE - 1:  # 's' for down
        agent_position[0] += 1
    elif direction == 'a' and agent_position[1] > 0:  # 'a' for left
        agent_position[1] -= 1
    elif direction == 'd' and agent_position[1] < GRID_SIZE - 1:  # 'd' for right
        agent_position[1] += 1
    else:
        print("Invalid move")

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
        print("going right")
    else:
        print("Invalid move")
        
    """
    In this code, the reward function is implicitly defined within the perform_action function.
    Specifically, it's the part where the code checks if the agent's position is at the reward location:

    The agent receives a reward of 7 if it reaches the position [1,4] on the grid,
    and a reward of 0 otherwise. This is effectively the reward function.
    """

    # Check if the agent has reached the reward position
    if agent_position == [1,4]:
        REWARDS += 7
        # Transport the agent to a random position within the grid after receiving the reward.
        agent_position[0] = random.randint(0, GRID_SIZE - 1)
        agent_position[1] = random.randint(0, GRID_SIZE - 1)
        return 7
    else:
        return 0

update_grid()
#print initial grid
print("INITIAL GRID WITH AGENT 8")
print("Your current Rewards =", REWARDS)
for row in grid:
    print(row)
print("------------------")

# Initialize the Q-table with zeros
Q_table = np.zeros((GRID_SIZE * GRID_SIZE, 4))  # 4 for the four possible actions: up, down, left, right

#print inital Q-Table
#print(Q_table)

# Define the learning rate and the discount factor
alpha = 0.5
gamma = 0.9

# Define a function to get the state index from the agent's position
def get_state_index(agent_position):
    return agent_position[0] * GRID_SIZE + agent_position[1]

# Define a function to choose the best action for a given state
def choose_best_action(state):
    # With probability epsilon, take a random action
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(4)  # 4 is the number of actions
    # Otherwise, take the action that has the highest Q-value
    else:
        action = np.argmax(Q_table[state])
    return action

# Update the Q-table using the Q-learning formula
def update_Q_table(prev_state, action, reward, next_state):
    Q_table[prev_state][action] = (1 - alpha) * Q_table[prev_state][action] + alpha * (reward + gamma * np.max(Q_table[next_state]))

"""
state = get_state_index(agent_position)
print(state)
action = choose_best_action(state)
print(action)
reward = perform_action(action)
print(reward)
next_state = get_state_index(agent_position)
print(next_state)
update_Q_table(state, action, reward, next_state)
print(Q_table)
"""


# In your main loop, use the Q-table to choose the best action and update the Q-table

# Number of iterations
N_ITERATIONS = 1000000

# In your main loop, use the Q-table to choose the best action and update the Q-table
for i in range(N_ITERATIONS):
    print(f"Current iteration: {i+1}")

    # Get the current state
    state = get_state_index(agent_position)

    # Choose the best action
    action = choose_best_action(state)

    # Perform the action and get the reward
    reward = perform_action(action)

    # Get the next state
    next_state = get_state_index(agent_position)

    # Update the Q-table
    update_Q_table(state, action, reward, next_state)

# Save the final Q-table
np.save('final_Q_table.npy', Q_table)


"""
while True:
    # Get user input and move the agent
    user_input = input("Enter direction (w for up, s for down, a for left, d for right) or 'q' to quit: ")
    if user_input == 'q':
        break

    if agent_position == [1,4]:
        REWARDS += 7
        # Transport the agent to a random position within the grid
        agent_position[0] = random.randint(0, GRID_SIZE - 1)
        agent_position[1] = random.randint(0, GRID_SIZE - 1)
    

    move_agent(user_input)

    update_grid()

    # Print the updated grid and reward status 
    print("Your current Rewards =", REWARDS)
    for row in grid:
        print(row)
"""
        


"""
# Function to move the agent
def move_agent(direction):
    if direction == 'up' and agent_position[0] > 0:
        agent_position[0] -= 1
    elif direction == 'down' and agent_position[0] < GRID_SIZE - 1:
        agent_position[0] += 1
    elif direction == 'left' and agent_position[1] > 0:
        agent_position[1] -= 1
    elif direction == 'right' and agent_position[1] < GRID_SIZE - 1:
        agent_position[1] += 1
    else:
        print("Invalid move")


#hardcoded agent movements
#Move the agen in the grid up
move_agent('w')
print(agent_position)  # Prints: [0, 0] because the agent can't move up from the top row
#Move the agen in the grid down
move_agent('s')
print(agent_position)
#Move the agen in the grid down 
move_agent('s')
print(agent_position)

update_grid()


#print the upgraded grid
for row in grid:
    print(row)
"""






