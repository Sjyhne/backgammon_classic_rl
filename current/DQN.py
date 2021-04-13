import gym
import numpy as np
from tqdm import tqdm
from numpy import load, asarray, save, savetxt
import matplotlib.pyplot as plt
from tensorforce import Agent, Environment
from agents import RandomAgent


def run_game(env, agents, render=False):
    WHITE = 0
    BLACK = 1
    winner, done = None, False

    rewards = 0
    rounds = 0

    done = False

    if render:
        env.render()
    # print(env.current_agent)
    while not done:

        reward = 0
        if env.current_agent == WHITE:
            _, done, winner = agents[env.current_agent].apply_random_action(
                env)
        else:

            n_actions = env.get_n_actions()
            for _ in range(n_actions):
                executed = False
                all_actions = env.get_actions()
                for _ in range(len(all_actions)):
                    states = list(env.get_current_observation())

                    actions = agents[env.current_agent].act(states=states)
                    obs, reward, done, winner, executed = env.step(
                        action=tuple(actions)
                    )
                    rewards += reward

                    agents[env.current_agent].observe(
                        terminal=executed, reward=reward)
                    if executed:
                        # print(actions)
                        break

        env.change_player_turn()
        if render:
            env.render()
        rounds += 1

        if done:
            return winner, rewards
            break


env = gym.make("reduced_backgammon_gym:reducedBackgammonGym-v0")

agent = Agent.create(
    agent="tensorforce",
    states=dict(type="int", shape=11, num_values=9),
    actions=dict(type="int", shape=2, num_values=8),
    memory=10000,
    update=dict(unit="timesteps", batch_size=64),
    optimizer=dict(type="adam", learning_rate=3e-4),
    policy=dict(network="auto"),
    objective="policy_gradient",
    reward_estimation=dict(horizon=20),
)
agents = {0: RandomAgent(), 1: agent}


def run():
    WHITE = 0
    BLACK = 1
    episodes = 100
    wins = {WHITE: [], BLACK: []}
    rewards = []

    for _, episode in tqdm(enumerate(range(episodes))):
        # print(episode)
        env.reset()
        winner, reward = run_game(env, agents)
        # print("\n")
        if winner == BLACK:
            #print("Black wins")
            wins[BLACK].append(1)
            wins[WHITE].append(0)
        elif winner == WHITE:
            #print("White wins")
            wins[BLACK].append(0)
            wins[WHITE].append(1)
        else:
            wins[BLACK].append(0)
            wins[WHITE].append(0)
            print("DRAW/TIMEOUT", "...", winner)
        rewards.append(reward)
        # print(reward)

    print(
        f"WHITE WON {round(sum(wins[WHITE])/(sum(wins[WHITE]) + sum(wins[BLACK])), 2)*100}%")
    print(
        f"BLACK WON {round(sum(wins[BLACK])/(sum(wins[WHITE]) + sum(wins[BLACK])), 2)*100}%")
    print(rewards)


if __name__ == "__main__":
    run()
