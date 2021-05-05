# Pip installed imports
import gym
import numpy as np
from tqdm import tqdm
from numpy import load, asarray, save, savetxt
import matplotlib.pyplot as plt
from multiprocessing import shared_memory, Process, Queue, set_start_method, Lock

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
    # This must be updated because the original agent does not get updated when
    # The decay call is called inside of the process....., prob needs to be a shared memory..?
    # This workaround is best
    agents[BLACK].decay_epsilon(episode, episodes)

    agents[BLACK].Q_actions = 0
    agents[BLACK].random_actions = 0

    if render:
        print("Current agent:", env.current_agent)
        env.render()

    while not done:
        reward = 0

        if env.current_agent == WHITE:
            _, done, winner, _ = agents[env.current_agent].apply_random_action(env)
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
            elif winning_agent == BLACK:
                # Winning part
                new_q = agents[winning_agent].calculate_new_q(agents[winning_agent].last_observations[0], agents[winning_agent].last_observations[1], agents[winning_agent].last_action, 1)
                agents[winning_agent].update_Q(agents[winning_agent].last_observations[0], agents[winning_agent].last_action, new_q)

            if winning_agent == WHITE:
                agents[BLACK].wins.append(0)
            else:
                agents[BLACK].wins.append(1)

            queue.put({"winner": winner, "rewards": rewards, "rounds": rounds, "epsilon": agents[BLACK].epsilon, "unexperienced_states": agents[BLACK].unexperienced_states})
            return winner, rewards, rounds

    queue.put({"winner": winner, "rewards": rewards, "rounds": rounds, "epsilon": agents[BLACK].epsilon, "unexperienced_states": agents[BLACK].unexperienced_states})
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

    a = np.zeros((obs_space + a_space), dtype=np.float16)
    #a = load("./current/results/black.npy")
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)

    np_array = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)

    episodes = 20
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
    all_replayed_states = []
    replayed_states_count = []

    total_rounds = []
    results = []
    env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

    envs = [gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0') for i in range(4)]

    multiprocess = [np_array, shm]

    q_agents = [QAgent(obs_space, a_space, multiprocess=multiprocess) for i in range(4)]

    queue = Queue()
    eps = []
    wins = []

    process_count = 4

    agents = []
    for q in q_agents:
        agents.append({0: RandomAgent(), 1: q})

    tic = time.perf_counter()


    for _, i in tqdm(enumerate(range(episodes)), desc="Main loop"):

        replayed_states = []

        # Reset each environment
        for env in envs:
            env.reset()
            
        jobs = []

        for j in range(process_count):
            jobs.append(Process(target=run_game, args=[envs[j], i, episodes, agents[j], queue]))

        for job in jobs:
            job.start()

        while not queue.empty():
            res = queue.get()
            wins.append(res["winner"])
            rewards.append(res["rewards"])
            total_rounds.append(res["rounds"])
            eps.append(res["epsilon"])
            replayed_states.extend(res["unexperienced_states"])

        for job in jobs:
            job.join()

        replayed_states = list(set(replayed_states))
        replayed_states_count.append(len(replayed_states))

        if i % steps == 0 and i != 0:
            print()
            print("EPSILON        : ------------------------------------ ", round(q_agents[0].epsilon, 2))
            print("WIN BLACK RATIO: ------------------------------------ ", round(sum(wins[-steps:])/steps, 3))
            print("WIN WHITE RATIO: ------------------------------------ ", round(1 - sum(wins[-steps:])/steps, 3))


        print("Will now go through", len(replayed_states), "states")

        for _, state in tqdm(enumerate(replayed_states), desc="Unexp loop"):
            
            # Set each env to the specified state
            for env in envs:
                env.set_starting_state_and_player(state, BLACK)

            r_jobs = []

            for j in range(process_count):
                r_jobs.append(Process(target=run_game, args=[envs[j], i, episodes, agents[j], queue]))

            for job in r_jobs:
                job.start()

            while not queue.empty():
                res = queue.get()

            for job in r_jobs:
                job.join()
        


    toc = time.perf_counter()

    print("=====================================================================")
    print(f"=================== DONE TRAINING IN {round(toc - tic, 1)} SECONDS ===================")
    print("=====================================================================")
    print()

    #print("WIN RATIO:", wins[BLACK]/(wins[BLACK] + wins[WHITE]))

    player_wins = {WHITE: 0, BLACK: 0}
    p_wins = {WHITE: [], BLACK: []}

    for i in wins:
        if i == 1:
            player_wins[BLACK] += 1
            p_wins[WHITE].append(0)
            p_wins[BLACK].append(1)
        else:
            player_wins[WHITE] += 1
            p_wins[WHITE].append(1)
            p_wins[BLACK].append(0)


    print(f"WHITE WON {round(player_wins[WHITE]/(player_wins[WHITE] + player_wins[BLACK]), 2)}% out of {player_wins[WHITE] + player_wins[BLACK]} games")
    print(f"BLACK WON {round(player_wins[BLACK]/(player_wins[WHITE] + player_wins[BLACK]), 2)}% out of {player_wins[WHITE] + player_wins[BLACK]} games")

    print("NR OF UNEXPERIENCED STATES:", sum(replayed_states_count))
        #env.set_starting_state_and_player(i)

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
        try:
            save(npy_path, np_array)
        except Exception as e:
            print("Exception:", e)

        print(np_array[2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 1, 6].round(6) == q_agents[0].Q[2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 1, 6].round(6))
        print(np_array[2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 1, 6].round(6))

        save_list_to_file(wins, "wins.txt", path)

        save_list_to_file(rewards, "rewards.txt", path)
        save_list_to_file(total_rounds, "rounds.txt", path)
        save_list_to_file(list(set(eps)), "epsilons.txt", path)
        save_list_to_file(replayed_states_count, "replayed_states_coun.txt", path)
        #save_list_to_file(qs, "black_qs.txt", path)
        #save_list_to_file(Q_actions, "Q_actions.txt", path)
        #save_list_to_file(random_actions, "random_actions.txt", path)

        # Save plots to plots folder
        plot_path = os.path.join(path, "plots")
        os.mkdir(plot_path)

        single_plot(rewards, steps, "Rewards", plot_path)
        single_plot(total_rounds, steps, " Avg Rounds", plot_path)
        single_plot(eps, steps, "Epsilons", plot_path)
        single_plot(replayed_states_count, steps, "Replayed states", plot_path)
        dual_plot(p_wins, steps, "Wins", plot_path)
        #plot_actions(random_actions, Q_actions, steps, plot_path)
        #plot_qs(qs, plot_path)

        with open(path + "/info.txt", "w") as f:
            f.write(f"Epsilon: {q_agents[0].epsilon}, Discount: {q_agents[0].discount}, LR: {q_agents[0].lr}")

    shm.close()
    shm.unlink()
if __name__ == "__main__":
    run(saveFile=True)
