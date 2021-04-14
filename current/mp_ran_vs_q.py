# Pip installed imports
import gym
import numpy as np
from tqdm import tqdm
from numpy import load, asarray, save, savetxt
import matplotlib.pyplot as plt
from multiprocessing import shared_memory, Process, Queue

# Basic imports
import random
import math
import time
import os

from agents import RandomAgent, QAgent

# Selfmade imports
from utils import flip_observation, flip_action

def run_game(env, episode, episodes, agents, queue, render=False):
    WHITE = 0
    BLACK = 1   
    current_agent = env.current_agent
    winner, done = None, False

    rounds = 0

    rewards = 0

    agents[BLACK].Q_actions = 0
    agents[BLACK].random_actions = 0

    if render:
        print("Current agent:", env.current_agent)
        env.render()

    while not done:
        reward = 0

        if env.current_agent == WHITE:
            _, done, winner = agents[env.current_agent].apply_random_action(env)
        else:
            _, done, winner, rew = agents[env.current_agent].execute_action(env)
            rewards += rew

        env.change_player_turn()

        if render:
            print("Current agent:", env.current_agent)
            env.render()

        rounds += 1

        if rounds > 9995:
            env.render()

        if rounds > 10_000:
            print("ROUNDS MORE THAN 10 000")
            return -1, rewards, rounds

        if done:
            winning_agent = env.current_agent
            losing_agent = 0 if env.current_agent == 1 else 1
            if winning_agent == WHITE:
                # Losing part
                new_q = agents[losing_agent].calculate_new_q(agents[losing_agent].last_observations[0], agents[losing_agent].last_observations[1], agents[losing_agent].last_action, -1)
                agents[losing_agent].update_Q(agents[losing_agent].last_observations[0], agents[losing_agent].last_action, new_q)
                agents[losing_agent].decay_epsilon(episode, episodes)
            elif winning_agent == BLACK:
                # Winning part
                new_q = agents[winning_agent].calculate_new_q(agents[winning_agent].last_observations[0], agents[winning_agent].last_observations[1], agents[winning_agent].last_action, 1)
                agents[winning_agent].update_Q(agents[winning_agent].last_observations[0], agents[winning_agent].last_action, new_q)
                agents[winning_agent].decay_epsilon(episode, episodes)

            if winning_agent == WHITE:
                agents[BLACK].wins.append(0)
            else:
                agents[BLACK].wins.append(1)

            queue.put([winner, rewards, rounds, agents[BLACK].epsilon])

            return winner, rewards, rounds
    queue.put([winner, rewards, rounds, agents[BLACK].epsilon])
    return winner, rewards, rounds


def dual_plot(dict, steps, title="", plot_path="./"):
    WHITE = 0
    BLACK = 1

    white_wins = [sum(dict[WHITE][x:x + steps])/steps for x in range(0, len(dict[WHITE]), steps)]
    black_wins = [sum(dict[BLACK][x:x + steps])/steps for x in range(0, len(dict[BLACK]), steps)]

    plt.plot(white_wins, label="White")
    plt.plot(black_wins, label="Black")

    plt.legend()

    plt.title(title)

    plt.savefig(plot_path + "/" + title.lower())

    plt.close()

def single_plot(lst, steps, title="", plot_path="./"):
    lst = [sum(lst[x:x + steps])/steps for x in range(0, len(lst), steps)]
    plt.plot(lst)
    plt.title(title)
    plt.savefig(plot_path + "/" + title.lower())
    plt.close()

def plot_qs(qs, plot_path):

    plt.plot(qs, label="Black")
    plt.title("Q.sum()")
    plt.savefig(plot_path + "/" + "Q sum")
    plt.close()

def plot_actions(random_actions, q_actions, steps, plot_path):
    r_actions = [sum(random_actions[x:x + steps])/steps for x in range(0, len(random_actions), steps)]
    q_actions = [sum(q_actions[x:x + steps])/steps for x in range(0, len(q_actions), steps)]
    plt.plot(r_actions, label="Random Actions")
    plt.plot(q_actions, label="Q Actions")
    plt.legend()
    plt.title("Random vs. Q Actions")
    plt.savefig(plot_path + "/" + "Random vs Q Actions".lower())
    plt.close()

def save_list_to_file(lst, name, path):

    file_path = os.path.join(path, name)

    with open(file_path, "w") as f:
        for item in lst:
            f.write(f"{item}\n")

def run(saveFile=False):
    # Predefined variables
    obs_space = (9, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2)
    a_space = (8, 8)

    #a = np.zeros((obs_space + a_space), dtype=np.float16)
    a = load("./current/results/black.npy")
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    print(a[2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 1])

    episodes = 100
    steps = episodes//20
    WHITE = 0
    BLACK = 1
    COLORS = {WHITE: "White", BLACK: "Black"}
    # Define the agents
    print("Initiating agents")
    print("Successfully initiated the agents")
    # For plotting later
    rewards = []
    qs = []

    Q_actions = []
    random_actions = []

    replayed_states = []

    total_rounds = []
    results = []
    env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

    envs = [gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0') for i in range(4)]

    multiprocess = [a, shm]

    q_agents = [QAgent(obs_space, a_space, multiprocess=multiprocess, train=False, epsilon=0) for i in range(4)]

    queues = [Queue() for i in range(4)]
    eps = []
    wins = []

    agents = []
    for q in q_agents:
        agents.append({0: RandomAgent(), 1: q})
    

    for _, i in tqdm(enumerate(range(episodes))):
        p1 = Process(target=run_game, args=[envs[0], i, episodes, agents[0], queues[0]])
        p2 = Process(target=run_game, args=[envs[1], i, episodes, agents[1], queues[1]])
        p3 = Process(target=run_game, args=[envs[2], i, episodes, agents[2], queues[2]])
        p4 = Process(target=run_game, args=[envs[3], i, episodes, agents[3], queues[3]])

        p1.start()
        p2.start()
        p3.start()
        p4.start()

        for que in queues:
            res = que.get()
            wins.append(res[0])
            rewards.append(res[1])
            total_rounds.append(res[2])
            eps.append(res[3])

        p1.join()
        p2.join()
        p3.join()
        p4.join()


    print("DONE TRAINING")

    #print("WIN RATIO:", wins[BLACK]/(wins[BLACK] + wins[WHITE]))

    player_wins = {WHITE: 0, BLACK: 0}

    for i in wins:
        if i == 1:
            player_wins[BLACK] += 1
        else:
            player_wins[WHITE] += 1

    print("WIN RATIO:", player_wins[BLACK]/(player_wins[BLACK] + player_wins[WHITE]))

    shm.close()
    shm.unlink()

    """
    episodes = 10_000
    steps = episodes//20
    WHITE = 0
    BLACK = 1
    COLORS = {WHITE: "White", BLACK: "Black"}
    # Define the agents
    print("Initiating agents")
    agents = {WHITE: RandomAgent(), BLACK: QAgent(obs_space, a_space, load_path="./current/results/black.npy")}
    print("Successfully initiated the agents")
    # For plotting later
    wins = {WHITE: [], BLACK: []}
    rewards = []
    qs = []

    Q_actions = []
    random_actions = []

    replayed_states = []

    total_rounds = []
    result = []
    env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

    tic = time.perf_counter()

    for _, episode in tqdm(enumerate(range(episodes))):
        
        env.reset()

        winner, game_rewards, rounds = run_game(env, episode, episodes, agents, render=False)

        if len(set(agents[1].unexperienced_states)) > 0:
            print(f"--------------- REPLAYING {len(set(agents[1].unexperienced_states))} STATES ---------------")
            for _ in range(50):
                for i in list(set(agents[1].unexperienced_states)):
                    env.set_starting_state_and_player(i)
                    run_game(env, episode, episodes, agents, render=False)

        replayed_states.append(len(set(agents[1].unexperienced_states)))
        
        agents[1].unexperienced_states = []

        Q_actions.append(agents[BLACK].Q_actions)
        random_actions.append(agents[BLACK].random_actions)

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

        rewards.append(game_rewards)

        total_rounds.append(rounds)

        if winner != None:
            result.append(winner)

        if episode % steps == 0 and episode != 0:
            print()
            print("EPSILON        : ------------------------------------ ", round(agents[BLACK].epsilon, 2))
            print("WIN BLACK RATIO: ------------------------------------ ", round(sum(result[-steps:])/steps, 3))
            print("WIN WHITE RATIO: ------------------------------------ ", round(1 - sum(result[-steps:])/steps, 3))

        if episode % (episodes//15) == 0 and episode != 0:
            qs.append(np.absolute(agents[BLACK].Q.sum()))


    toc = time.perf_counter()

    print(f"============EPISODES ARE DONE IN {round(toc - tic, 1)} SECONDS============")

    print(f"WHITE WON {round(sum(wins[WHITE])/(sum(wins[WHITE]) + sum(wins[BLACK])), 2)}%")
    print(f"BLACK WON {round(sum(wins[BLACK])/(sum(wins[WHITE]) + sum(wins[BLACK])), 2)}%")

    for i in list(set(agents[1].unexperienced_states)):
        env.set_starting_state_and_player(i)

    # print(agents[BLACK].Q[2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 1])

    #if saveFile and agents[BLACK].train:
        #save(f"q_tables/new_q_{episodes}_{steps}_white.npy", agents[WHITE].Q)
        #save(f"q_tables/ran_v_q_{episodes}_{steps}_black.npy", agents[BLACK].Q)

    # plot_qs(qs)

    if saveFile:
        directory = f"ran_v_q_{episodes}_{steps}_black"
        parent_directory = "./current/results"
        path = os.path.join(parent_directory, directory)
        os.mkdir(path)

        npy_path = os.path.join(path, "black.npy")
        save(npy_path, agents[BLACK].Q)

        save_list_to_file(wins[WHITE], "white_wins.txt", path)
        save_list_to_file(wins[BLACK], "black_wins.txt", path)

        save_list_to_file(rewards, "rewards.txt", path)
        save_list_to_file(total_rounds, "rounds.txt", path)
        save_list_to_file(agents[BLACK].epsilons, "epsilons.txt", path)
        save_list_to_file(qs, "black_qs.txt", path)
        save_list_to_file(Q_actions, "Q_actions.txt", path)
        save_list_to_file(random_actions, "random_actions.txt", path)

        # Save plots to plots folder
        plot_path = os.path.join(path, "plots")
        os.mkdir(plot_path)

        single_plot(rewards, steps, "Rewards", plot_path)
        single_plot(total_rounds, steps, " Avg Rounds", plot_path)
        single_plot(agents[BLACK].epsilons, steps, "Epsilons", plot_path)
        single_plot(replayed_states, steps, "Replayed states", plot_path)
        dual_plot(wins, steps, "Wins", plot_path)
        plot_actions(random_actions, Q_actions, steps, plot_path)
        plot_qs(qs, plot_path)

        with open(path + "/info.txt", "w") as f:
            f.write(f"Epsilon: {agents[BLACK].epsilon}, Discount: {agents[BLACK].discount}, LR: {agents[BLACK].lr}")
    """


if __name__ == "__main__":
    run(saveFile=True)
