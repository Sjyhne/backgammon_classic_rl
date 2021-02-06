import gym
from itertools import count
import numpy as np
import random
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN


"""
    WHITE = 0
    BLACK = 1
    COLORS = {0: WHITE, 1: BLACK}

"""

class q_agent:
    def __init__(self, color):
        self.color = color
        self.name = f"{COLORS[self.color]}_QAgent"

    def roll_dice(self):
        if self.color == WHITE:
            return (-random.randint(1, 6), -random.randint(1, 6))
        else:
            return (random.randint(1, 6), random.randint(1, 6))

    def choose_best_action(self, actions, env):
        if self.color == WHITE:
            return list(actions)[0] if actions else None
        else:
            if actions:
                if len(list(actions)) >= 2:
                    return list(actions)[1]
                elif len(list(actions)) == 1:
                    return list(actions)[0]
            else:
                return None

    
    def __str__(self):
        return self.name

def initiate_agents():
    white = q_agent(WHITE)
    black = q_agent(BLACK)
    agents = {WHITE: white, BLACK: black}
    return agents

def play_game():

    # Create the gym for the agents
    env = gym.make('gym_backgammon:backgammon-v0')

    # Initiate the agents of the game
    agents = initiate_agents()

    starting_color, first_roll, observation = env.reset()

    agent = agents[starting_color]

    env.render(mode="human")

    """
    print("FIRST ROLL:", first_roll)
    print("WHITE:", WHITE)
    print("BLACK:", BLACK)
    print("COLORS:", COLORS)
    print("CURRENT PLAYER:", COLORS[starting_color])
    print("TOKENS:", TOKEN)
    """

    for i in count():
        if first_roll:
            roll = first_roll
            first_roll = None
        else:
            roll = agent.roll_dice()
        
        print("CURRENT AGENT:", agent)
        print("DICE THROW:", roll)
        valid_actions = env.get_valid_actions(roll)
        print("POSSIBLE_ACTIONS:", valid_actions)
        action = agent.choose_best_action(valid_actions, env)
        print("CHOSEN ACTION:", action)
        next_observation, reward, done, winner = env.step(action)
        agent_color = env.get_opponent_agent()
        agent = agents[agent_color]

        env.render()

        if done:
            print("DONE")
            print("NUMBER OF TURNS:", i)
            break

if __name__ == "__main__":
    play_game()