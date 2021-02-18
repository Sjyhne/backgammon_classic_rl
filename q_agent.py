import gym
from itertools import count
import numpy as np
import random
from backgammon_simplified.envs.s_backgammon import COLORS, TOKEN, WHITE, BLACK
from backgammon_simplified.envs.s_backgammon_env import SimplifiedBackgammonEnv

"""
    Must make the agent able to choose illegal actions, and
    then returning a reward based on the whether the action is
    legal or not

    I'm not sure whether the agent should lose it's turn when
    it chooses an illegal action?
"""

class RandomAgent:
    def __init__(self, color):
        self.color = color
        self.name = COLORS[self.color]

    def dice_roll(self):
        return (-random.randint(1, 3), -random.randint(1, 3)) if self.color == WHITE else (random.randint(1, 3), random.randint(1, 3))

    def choose_best_action(self, actions, obs):
        return random.choice(list(actions)) if actions else None


def play_game():

    env = gym.make("backgammon_simplified:backgammon-v69")

    agent, roll, observation = env.reset()

    agents = {WHITE: RandomAgent(WHITE), BLACK: RandomAgent(BLACK)}

    agent_color, first_roll, observation = env.reset()

    agent = agents[agent_color]

    env.render()

    for i in count():
        if first_roll:
            roll = first_roll
            first_roll = None
        else:
            roll = agent.dice_roll()

        valid_actions = env.get_valid_actions(roll)

        all_actions = env.get_all_actions(roll)

        action = agent.choose_best_action(valid_actions, observation)

        next_obseration, reward, done, winner = env.step(action)
        env.render()

        if done:
            if winner != None:
                print(COLORS[winner], "WON AFTER", i, "ROUNDS!")
                exit(1)

        agent_color = env.get_opponent_agent()
        agent = agents[agent_color]
        observation = next_obseration

play_game()