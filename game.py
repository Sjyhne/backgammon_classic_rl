import gym
import random

env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

import numpy as np
from numpy import load

done = False

obs, current_agent = env.reset()

BLACK = 1
WHITE = 0

COLORS = {WHITE: "White", BLACK: "Black"}

from current.agents import RandomAgent, QAgent


env.render()

obs_space = (9, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2)
a_space = (8, 8)

print("Loading Agent")
q_agent = QAgent(obs_space, a_space, load_path="black.npy", epsilon=0, train=False)
print("Successfullt loaded Agent")

while not done:

    print("Current agent:", COLORS[env.current_agent])

    if env.current_agent == BLACK:
        obs, done, winner, rew = q_agent.execute_action(env)
        env.render()

    else:
        for _ in range(env.get_n_actions()):

            available_actions = env.get_valid_actions()

            if len(available_actions) == 0:
                break
            
            print(env.gym.non_used_dice)
            for idx, i in enumerate(available_actions):
                print(idx + 1, ":", i)

            action_index = int(input("Input index of preferred action"))

            action = available_actions[action_index - 1][1]

            obs, rew, done, winner, executed = env.step(action)

            env.render()

    env.change_player_turn()

    



