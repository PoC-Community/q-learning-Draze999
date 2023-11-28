import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt



# Initializing an empty table

def init_q_table(x: int, y: int) -> np.ndarray:
    """
    This function must return a 2D matrix containing only zeros for values.
    """
    return np.zeros((x, y))

# qTable = init_q_table(5, 4)

# print("Q-Table:\n" + str(qTable))

# assert(np.mean(qTable) == 0)







LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.99

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, newState: int) -> float:
    """
    This function must implement the q_function equation pictured above.

    It should return the updated q-table.
    """

    number = q_table[state, action] + LEARNING_RATE * (reward + DISCOUNT_RATE * max(q_table[newState, :] - q_table[state, action]))
    return number

# q_table = init_q_table(5,4)

# q_table[0, 1] = q_function(q_table, state=0, action=1, reward=-1, newState=3)

# print("Q-Table after action:\n" + str(q_table))

# assert(q_table[0, 1] == -LEARNING_RATE), f"The Q function is incorrect: the value of qTable[0, 1] should be -{LEARNING_RATE}"







# map_name1:[
#     "FFBFFF",
#     "FHFSHF",
#     "FFBFFB",
#     "HFFHFF",
#     "FBFFGF"
#     ]
# map_name2:[
#     "FFFFFF",
#     "FHFSHF",
#     "FFFFFF",
#     "HFFHFF",
#     "FFFFGF"
#     ]

# env = gym.make('FrozenLake-v1', map_name="4x4", render_mode="rgb_array", is_slippery=False)
# env.reset()
# env.render()


# total_actions = env.action_space.n
# assert(total_actions == 4), f"There are a total of four possible actions in this environment. Your answer is {total_actions}"

def random_action(env):
    return env.action_space.sample()

# observation, info = env.reset()

# # Performing an action
# action = random_action(env)
# observation, reward, done, _, info = env.step(action)

# # Displaying the first frame of the game
# plt.imshow(env.render())

# # Printing game info
# # print(f"actions: {env.action_space.n}\nstates: {env.observation_space.n}")
# # print(f"Current state: {observation}")

# # Closing the environment
# env.close()



def game_loop(env: gym.Env, q_table: np.ndarray, state: int, action: int) -> tuple:
    newState, reward, done, _, info = env.step(action)
    q_table[state, action] = q_function(q_table, state, action, reward, newState)
    return q_table, newState, done, reward

# env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
# q_table = init_q_table(env.observation_space.n, env.action_space.n)

# state, info = env.reset()
# while (True):
#     env.render()
#     action = random_action(env)
#     q_table, state, done, reward = game_loop(env, q_table, state, action)
#     if done:
#         break
# env.close()





EPOCH = 20000

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
q_table = init_q_table(env.observation_space.n, env.action_space.n)

for i in range(EPOCH):
    state, info = env.reset()
    while (True):
        # This time, we won't render the game each frame because it would take too long
        action = random_action(env)
        q_table, state, done, reward = game_loop(env, q_table, state, action)
        if done:
            break
env.close()

# # Printing the QTable result:
# for states in q_table:
#     for actions in states:
#         if (actions == max(states)):
#             print("\033[4m", end="")
#         else:
#             print("\033[0m", end="")
#         if (actions > 0):
#             print("\033[92m", end="")
#         else:
#             print("\033[00m", end="")
#         print(round(actions, 3), end="\t")
#     print()

def best_action(q_table: np.ndarray, state: int) -> int:
    """
    Write a function which finds the best action for the given state.

    It should return its index.
    """
    return np.argmax(q_table[state, :])

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

state, info = env.reset()
while (True):
    env.render()
    action = best_action(q_table, state)
    q_table, state, done, reward = game_loop(env, q_table, state, action)
    if done:
        break

env.close()