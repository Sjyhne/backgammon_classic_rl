# Pip installed imports
import gym
import numpy as np
from tqdm import tqdm
from numpy import load, asarray, save, savetxt
import matplotlib.pyplot as plt

# Basic imports
import random
import math
import time

from agents import RandomAgent, QAgent, QAgentFlip

from utils import flip_action, flip_observation

"""
    Hvor må ting flippes?

    1. Jeg får inn observation som IKKE er flipped
        Den må flippes 
"""

def run_game(env, episode, episodes, agents, render=False):
    WHITE = 0
    BLACK = 1
    _, current_agent = env.reset()
    winner, done = None, False

    rounds = 0

    rewards = {WHITE: 0, BLACK: 0}

    if render:
        print("Current agent:", env.current_agent)
        env.render()

    while not done:
        reward = 0

        if env.current_agent == BLACK:
            _, done, winner, rew = agents[env.current_agent].execute_action(env)
        else:
            _, done, winner, rew = agents[env.current_agent].execute_action(env, flip=True)

        env.change_player_turn()

        if render:
            print("Current agent:", env.current_agent)
            env.render()

        rounds += 0
        rewards[env.current_agent] += rew

        if rounds > 9995:
            env.render()

        if rounds > 9_000:
            print("ROUNDS MORE THAN 9 000")
            return -1, rewards, rounds

        if done:
            winning_agent = env.current_agent
            losing_agent = 0 if env.current_agent == 1 else 1
            # Losing part
            new_q = agents[losing_agent].calculate_new_q(agents[losing_agent].last_observations[0], agents[losing_agent].last_observations[1], agents[losing_agent].last_action, -1)
            agents[losing_agent].update_Q(agents[losing_agent].last_observations[0], agents[losing_agent].last_action, new_q)
            agents[losing_agent].decay_epsilon(episode, episodes)
            # Winning part
            new_q = agents[winning_agent].calculate_new_q(agents[winning_agent].last_observations[0], agents[winning_agent].last_observations[1], agents[winning_agent].last_action, 1)
            agents[winning_agent].update_Q(agents[winning_agent].last_observations[0], agents[winning_agent].last_action, new_q)
            agents[winning_agent].decay_epsilon(episode, episodes)

            return winner, rewards, rounds

    return winner, rewards, rounds


def dual_plot(dict, steps, title=""):
    WHITE = 0
    BLACK = 1

    white_wins = [sum(dict[WHITE][x:x + steps])/steps for x in range(-1, len(dict[WHITE]), steps)]
    black_wins = [sum(dict[BLACK][x:x + steps])/steps for x in range(-1, len(dict[BLACK]), steps)]

    plt.plot(white_wins, label="White")
    plt.plot(black_wins, label="Black")

    plt.legend()

    plt.title(title)

    plt.show()

    ...

def single_plot(lst, steps, title=""):
    lst = [sum(lst[x:x + steps])/steps for x in range(-1, len(lst), steps)]
    plt.plot(lst)
    plt.title(title)
    plt.show()

def plot_qs(qs):
    WHITE = 0
    BLACK = 1

    white_qs = qs[WHITE]
    black_qs = qs[BLACK]

    plt.plot(white_qs, label="White")
    plt.plot(black_qs, label="Black")

    plt.legend()

    plt.title("abs(Q.sum())")
    plt.show()


def dual_action_plot(white_actions, black_actions):
    f, subplot = plt.subplots(2)
    subplot[0].imshow(white_actions, cmap="hot", interpolation="nearest")
    subplot[1].imshow(black_actions, cmap="hot", interpolation="nearest")
    subplot[0].set_title("White")
    subplot[1].set_title("Black")

    plt.show()

def run(saveFiles=False):
    # Predefined variables
    obs_space = (9, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2)
    a_space = (8, 8)
    episodes = 10_000
    steps = episodes//20
    WHITE = 0
    BLACK = 1
    COLORS = {WHITE: "White", BLACK: "Black"}
    # Define the agents
    print("Initiating agents")
    agent = QAgentFlip(obs_space, a_space)
    agents = {WHITE: agent, BLACK: agent}
    print("Successfully initiated the agents")
    # For plotting later
    wins = {WHITE: [], BLACK: []}
    rewards = {WHITE: [], BLACK: []}
    qs = {WHITE: [], BLACK: []}

    total_rounds = []
    result = []
    env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

    tic = time.perf_counter()

    for _, episode in tqdm(enumerate(range(episodes))):

        winner, game_rewards, rounds = run_game(env, episode, episodes, agents)

        if winner == BLACK:
            wins[BLACK].append(1)
            wins[WHITE].append(0)
        elif winner == WHITE:
            wins[BLACK].append(0)
            wins[WHITE].append(1)
        else:
            wins[BLACK].append(0)
            wins[WHITE].append(0)
            print("DRAW/TIMEOUT", "...", winner)

        rewards[BLACK].append(game_rewards[BLACK])
        rewards[WHITE].append(game_rewards[WHITE])

        total_rounds.append(rounds)

        if winner != None:
            result.append(winner)

        if episode % steps == 0 and episode != 0:
            print()
            print("WIN BLACK RATIO: ------------------------------------ ", sum(result[-steps:])/steps)
            print("WIN WHITE RATIO: ------------------------------------ ", 1 - sum(result[-steps:])/steps)

        # if episode % (episodes//5) == 0 and episode != 0:
        #     qs[WHITE].append(np.absolute(agents[WHITE].Q).sum())
        #     qs[BLACK].append(np.absolute(agents[BLACK].Q).sum())


    toc = time.perf_counter()

    print(f"============EPISODES ARE DONE IN {round(toc - tic, 1)} SECONDS============")

    print(f"WHITE WON {round(sum(wins[WHITE])/(sum(wins[WHITE]) + sum(wins[BLACK])), 2)}%")
    print(f"BLACK WON {round(sum(wins[BLACK])/(sum(wins[WHITE]) + sum(wins[BLACK])), 2)}%")

    if saveFiles == True:
        save(f"q_tables/self_v_q_{episodes}_{steps}_black.npy", agents[BLACK].Q)
        #save(f"q_tables/duo_q_{episodes}_{steps}_black.npy", agents[BLACK].Q)


    dual_plot(wins, steps, "Wins")
    dual_plot(rewards, steps, "Rewards")
    single_plot(total_rounds, steps, " Avg Rounds")
    dual_action_plot(agents[0].actions_executed, agents[1].actions_executed)



if __name__ == "__main__":
    run(saveFiles=True)
