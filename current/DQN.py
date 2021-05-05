import gym
import numpy as np
from tqdm import tqdm
from numpy import load, asarray, save, savetxt
import matplotlib.pyplot as plt
from tensorforce import Agent, Environment
from agents import RandomAgent

from utils import floatify_obs


def run_game(env, agents, render=False, eval=False):
    WHITE = 0
    BLACK = 1
    winner, done = None, False

    n_a = {WHITE: 0, BLACK: 0}

    rewards = 0
    rounds = 0
    tried_moves = []

    done = False

    if render:
        env.render()
    # print(env.current_agent)
    while not done:

        reward = 0
        if env.current_agent == WHITE:
            _, done, winner, n_actions = agents[env.current_agent].apply_random_action(
                env
            )
            n_a[WHITE] += n_actions
        else:

            n_actions = env.get_n_actions()
            for _ in range(n_actions):
                n_moves = 0
                executed = False
                all_actions = env.get_actions()
                states = floatify_obs(list(env.get_current_observation()))
                if eval:
                    actions = agents[env.current_agent].act(states=states)
                    obs, reward, done, winner, executed = env.step(
                        action=tuple(actions)
                    )
                    agents[env.current_agent].observe(terminal=executed, reward=reward)
                else:
                    for i in range(70):
                        n_moves += 1
                        actions = agents[env.current_agent].act(states=states)
                        obs, reward, done, winner, executed = env.step(
                            action=tuple(actions)
                        )
                        rewards += reward

                        agents[env.current_agent].observe(
                            terminal=executed, reward=reward
                        )
                        if executed:
                            n_a[BLACK] += 1
                            tried_moves.append(n_moves)
                            break

        env.change_player_turn()

        if render:
            env.render()
        rounds += 1

        if done:
            return winner, rewards, rounds, n_a, tried_moves
            break


env = gym.make("reduced_backgammon_gym:reducedBackgammonGym-v0")

agent = Agent.create(
    agent="tensorforce",
    states=dict(type="float", shape=11, max_value=0.8, min_value=0.0),
    actions=dict(type="int", shape=2, num_values=8),
    memory=dict(capacity=10000),
    update=dict(unit="timesteps", batch_size=16),
    optimizer=dict(type="adam", learning_rate=3e-4),
    policy=dict(network="auto"),
    objective="policy_gradient",
    reward_estimation=dict(horizon=20, discount=0.9),
    exploration=dict(
        type="linear",
        unit="episodes",
        num_steps=100,
        initial_value=0.9,
        final_value=0.8,
    ),
    config=dict(device="GPU:0"),
)
agents = {0: RandomAgent(), 1: agent}


def run():
    WHITE = 0
    BLACK = 1
    episodes = 5000
    results = []
    rewards = []
    rounds = []
    tried_moves = []
    wins_per_50 = []
    tmp_wins = 0
    eval_wins = []

    action_count = {WHITE: [], BLACK: []}

    for _, episode in tqdm(enumerate(range(episodes))):
        # print(episode)
        env.reset()
        winner, reward, r, n, tried_moves = run_game(env, agents)
        if winner == BLACK:
            winner += 1
        if episode % 50 == 0 and episode != 0:
            wins_per_50.append(tmp_wins)
            for i in tqdm(enumerate(range(50))):
                winner, reward, r, n, tried_moves = run_game(env, agents)
                eval_wins.append(winner)
                print("eval winner" + str(winner))

        # print(winner)
        rounds.append(r)
        # print("\n")
        results.append(winner)
        rewards.append(reward)
        # print(reward)
        action_count[WHITE].append(n[WHITE])
        action_count[BLACK].append(n[BLACK])

    steps = episodes // 10

    wins = {WHITE: 0, BLACK: 0}
    for i in results:
        if i == WHITE:
            wins[WHITE] += 1
        else:
            wins[BLACK] += 1

    try:
        print(f"WHITE WON {round(wins[WHITE]/(wins[WHITE] + wins[BLACK]), 2)*100}%")
        print(f"BLACK WON {round(wins[BLACK]/(wins[WHITE] + wins[BLACK]), 2)*100}%")
        # print([rewards[i] / rounds[i] for i in range(episodes)])

        print(
            f"Avg number of white moves per episode: {sum(action_count[WHITE]) / len(action_count[WHITE])}"
        )
        print(
            f"Avg number of black moves per episode: {sum(action_count[BLACK]) / len(action_count[BLACK])}"
        )
        print(f"Avg tries to find valid moves: {sum(tried_moves)/len(tried_moves)}")
        print(sum(eval_wins))
        print((len(eval_wins) - sum(eval_wins)) * -1)
        # print(lst2)
        plt.plot(action_count[WHITE], label="WHITE")
        plt.plot(action_count[BLACK], label="BLACK")
        # plt.plot(lst, label="BLACK")
        plt.plot(wins_per_50, label="Tried moves")
        plt.legend()
        plt.show()

        plt.plot(tmp_wins, label="Wins per 50 episode")
        plt.show()
    except RuntimeError:
        print("Something went wrong")


if __name__ == "__main__":
    run()

    """
    policy=dict(
        network=[
            dict(type="dense", size=16, activation="softmax"),
            dict(type="dense", size=16, activation="softmax"),
        ]
    ),
    """
