import random
import gymnasium as gym
import numpy as np
import matplotlib as plt

# Initializing an empty table

def init_q_table(x: int, y: int) -> np.ndarray:
    """
    This function must return a 2D matrix containing only zeros for values.
    """
    return np.zeros((x, y))

qTable = init_q_table(5, 4)

print("Q-Table:\n" + str(qTable))

assert(np.mean(qTable) == 0)

LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.99

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, newState: int) -> float:
    """
    This function must implement the q_function equation pictured above.

    It should return the updated q-table.
    """

    number = q_table[state, action] + LEARNING_RATE * (reward + DISCOUNT_RATE * max(q_table[newState, :] - q_table[state, action]))
    return number

q_table = init_q_table(5,4)

q_table[0, 1] = q_function(q_table, state=0, action=1, reward=-1, newState=3)

print("Q-Table after action:\n" + str(q_table))

assert(q_table[0, 1] == -LEARNING_RATE), f"The Q function is incorrect: the value of qTable[0, 1] should be -{LEARNING_RATE}"