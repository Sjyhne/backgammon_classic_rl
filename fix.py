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

# Selfmade imports
from utils import flip_observation, flip_action



class QAgent():

    def __init__(self, obs_space, action_space, lr=0.0001, discount=0.95, epsilon=1, load_path=None):

        # Agent configuration
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.load_path = load_path

        # Agent Q Table
        self.Q = self.initiate_Q(obs_space, action_space, load_path)

        # Last action and observations
        self.last_action = ()
        self.last_observations = []

        # Agent logging
        self.rewards = []
        self.wins = []
        self.knockouts = 0
        self.epsilons = []
        self.actions_executed = np.zeros((8, 8), dtype=np.int16)

    def initiate_Q(self, obs_space, action_space, load_path):
        if load_path == None:
            return np.zeros((obs_space + action_space), dtype=np.float16)
        else:
            return load(load_path)

    def update_Q(self, obs, action, value):
        self.Q[(tuple(obs) + tuple(action))] = value

    def update_last_actions(self, last_action, last_observation, last_next_observation):
        self.last_action = last_action
        self.last_observation = [last_observation, last_next_observation]

    def get_best_action(self, obs):
        action = np.unravel_index(np.argmax(self.Q[obs], axis=None), self.Q[obs].shape)
        print("Action:", action)
        return action

    def get_random_action(self, actions):
        return random.choice(actions)

    def decay_epsilon(self, episode, episodes):
        # Logging
        self.epsilons.append(self.epsilon)
        # Update epsilon following a ^2 curve
        self.epsilon = abs(np.linspace(-1, 0, episodes)[episode]**2)

    def calculate_new_q(self, obs, next_obs, action, reward):
        # The best possible future Q value
        max_future_q = np.max(self.Q[next_obs])
        # The Q value for the current action
        current_q = self.Q[obs + action]
        # The new Q for this action in the given observation state
        new_q = current_q + self.lr * (reward + self.discount * max_future_q - current_q)
        # return the new Q value
        return new_q

    def step(self, obs, next_obs, reward, done, winner, executed, action, actions):
        self.rewards.append(reward)
        self.update_Q(obs, action, reward)
        # If the action wasnt executed then continue to loop through the actions
        if not executed:
            if action in actions:
                action.remove(actions)
            return False
        else:
            # Update the last action made
            self.update_last_actions(action, obs, next_obs)
            self.actions_executed[action] += 1
            return True

    def execute_action(self, env):

        obs = env.get_current_observation()
        done = False
        winner = None

        num_actions = env.get_n_actions()

        for i in range(num_actions):

            random_actions = [i[1] for i in env.get_valid_actions()]

            # Check if the agent should to random or best action
            if random.uniform(-1, 1) < self.epsilon:

                # Number of random actions
                n_actions = len(random_actions)

                for n in range(n_actions):
                    # Choose a random action
                    random_action = self.get_random_action(random_actions)


                    # Try to perform the random action
                    next_obs, reward, done, winner, executed = env.step(random_action)

                    if self.step(obs, next_obs, reward, done, winner, executed, random_action, random_actions):
                        break
                    else:
                        continue
            else:
                action = self.get_best_action(obs)

                next_obs, reward, done, winner, executed = env.step(action)

                if self.step(obs, next_obs, reward, done, winner, executed, action, random_actions):
                    break
                else:
                    continue

        return obs, done, winner, sum(self.rewards)


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

        _, done, winner, rew = agents[env.current_agent].execute_action(env)

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
            new_q = agents[losing_agent].calculate_new_q(agents[losing_agent].last_observation[0], agents[losing_agent].last_observation[1], agents[losing_agent].last_action, -1)
            agents[losing_agent].update_Q(agents[losing_agent].last_observation[0], agents[losing_agent].last_action, new_q)
            agents[losing_agent].decay_epsilon(episode, episodes)
            # Winning part
            new_q = agents[winning_agent].calculate_new_q(agents[winning_agent].last_observation[0], agents[winning_agent].last_observation[1], agents[winning_agent].last_action, 1)
            agents[winning_agent].update_Q(agents[winning_agent].last_observation[0], agents[winning_agent].last_action, new_q)
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
    f, subplot = plt.subplots(1, 2)
    subplot[0, 0].imshow(white_actions, cmap="hot", interpolation="nearest")
    subplot[0, 1].imshow(black_actions, cmap="hot", interpolation="nearest")

    plt.show()

def run(saveFiles=False):
    # Predefined variables
    obs_space = (9, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2)
    a_space = (8, 8)
    episodes = 40
    steps = episodes//10
    WHITE = 0
    BLACK = 1
    COLORS = {WHITE: "White", BLACK: "Black"}
    # Define the agents
    print("Initiating agents")
    agents = {WHITE: QAgent(obs_space, a_space), BLACK: QAgent(obs_space, a_space)}
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
        save(f"q_tables/new_q_{episodes}_{steps}_white.npy", agents[WHITE].Q)
        save(f"q_tables/new_q_{episodes}_{steps}_black.npy", agents[BLACK].Q)


    dual_plot(wins, steps, "Wins")
    dual_plot(rewards, steps, "Rewards")
    single_plot(total_rounds, steps, " Avg Rounds")
    plot_qs(qs)
    dual_action_plot(agents[0].actions_executed, agents[1].actions_executed)



if __name__ == "__main__":
    run(saveFiles=False)
