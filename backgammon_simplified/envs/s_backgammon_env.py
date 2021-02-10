import gym
from gym.spaces import Box
from backgammon_simplified.envs.s_backgammon import Simplified_Backgammon as Game, WHITE, BLACK, COLORS, get_opponent_color
from random import randint
import numpy as np

class SimplifiedBackgammonEnv(gym.Env):
    metadata = {'render.modes': 'human'}

    def __init__(self):
        self.game = Game()
        self.current_agent = None
        
        self.counter = 0
        self.max_length_episodes = 1000000000

    def step(self, action):
        self.game.execute_play(self.current_agent, action)

        observation = self.game.get_game_features(get_opponent_color(self.current_agent))

        reward = 0
        done = False
        winner = self.game.get_winner()

        if winner is not None or self.counter > self.max_length_episodes:
            if winner == WHITE:
                reward = 1
            done = True

        self.counter += 1

        return observation, reward, done, winner

    def reset(self):
        roll = randint(1, 3), randint(1, 3)

        while roll[0] == roll[1]:
            roll = randint(1, 3), randint(1, 3)
        
        if roll[0] > roll[1]:
            self.current_agent = WHITE
            roll = (-roll[0], -roll[1])
        else:
            self.current_agent = BLACK

        self.game = Game()
        self.counter = 0

        return self.current_agent, roll, self.game.get_game_features(self.current_agent)

    def render(self):
        return self.game.render(self.counter)

    def get_valid_actions(self, roll):
        return self.game.get_valid_plays(self.current_agent, roll)
    
    def get_opponent_agent(self):
        self.current_agent = get_opponent_color(self.current_agent)
        return self.current_agent