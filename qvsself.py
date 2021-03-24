import gym
import numpy as np
import random
from tqdm import tqdm
import time
import multiprocessing as mp

from utils import flip_observation, flip_action

import matplotlib.pyplot as plt

# save numpy array as npy file
from numpy import asarray
from numpy import save, savetxt


EPISODES = 2000     
STEPS = 20

class QAgent:
    
    def __init__(self, obs_space, action_space, lr=0.1, discount=0.3, epsilon=1):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon

        self.rewards = []
        
        self.start_epsilon_decay = 1
        self.end_epsilon_decay = EPISODES - EPISODES*0.1
        self.epsilon_decay_value = self.epsilon/(self.end_epsilon_decay - self.start_epsilon_decay)

        self.Q = self.initiate_Q(obs_space, action_space)
    
    
    def initiate_Q(self, obs_space, action_space):
        # Initiates the Q table using the obs space and the action space
        return(np.zeros((obs_space + action_space), np.float16))

    def update_Q(self, obs, action, value):
        # Updates the Q table
        self.Q[(obs + action)] = value

    def get_best_action_given_observation(self, obs):
        # Get the best action for the given observation
        action = np.unravel_index(np.argmax(self.Q[obs], axis=None), self.Q[obs].shape)

        return action

    def get_random_action_given_observation(self, actions):
        action = random.choice(actions)

        return action

    def decay_epsilon(self):
        self.epsilon -= self.epsilon_decay_value

    def get_new_q_value(self, obs, next_obs, reward, action):
        # The best possible future Q value
        max_future_q = np.max(self.Q[next_obs])
        # The Q value for the current action
        current_q = self.Q[obs + action]
        # The new Q for this action in the given observation state
        new_q = (1 - self.lr) * current_q + self.lr * (reward + self.discount * max_future_q)
        # return the new Q value
        return new_q

    def apply_action(self, environment, flipped, m_rounds):
        num_actions = environment.get_n_actions()
        executed = False
        obs = tuple(environment.gym.get_current_observation(environment.current_agent))

        observation = obs

        if flipped:
            obs = flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots)

        rew = 0

        # print("Current agent:", COLORS[environment.current_agent])
        # print("Current roll:", environment.gym.non_used_dice)
        # print("Flipped:", flipped)
        # print("Observation:", observation)
        # print(num_actions)

        for _ in range(num_actions):
            actions = [i[1] for i in environment.get_actions()]

            len_act = len(actions)

            #print(environment.get_valid_actions())
            c = 0

            for _ in range(len(actions)):

                if random.uniform(0, 1) < self.epsilon:
                    action = self.get_random_action_given_observation(actions)
                else:
                    action = self.get_best_action_given_observation(obs)
                    #print("BEST:", action)

                if flipped:
                    action = flip_action(action, environment.gym.n_spots)

                
                next_observation, reward, done, winner, executed = environment.step(action)

                

                if flipped:
                    next_observation = flip_observation(next_observation, environment.gym.n_pieces, environment.gym.n_spots)

                new_q_value = self.get_new_q_value(obs, next_observation, reward, action)

                if reward != 0:
                    self.update_Q(obs, action, new_q_value)

                if executed:
                    obs = next_observation
                    break
                else:
                    if action in actions:
                        actions.remove(action)
                        #print("REMOVED:", action)
                    c += 1

            if c == len_act:
                break

        
        self.epsilon -= self.epsilon_decay_value

        if flipped:
            obs = flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots)
        
        return obs, done, winner, rew
            
            


#env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

observation_space = (9, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2)
action_space = (8, 8) # 0, ..., 7

WHITE = 0
BLACK = 1

COLORS = {WHITE: "White", BLACK: "Black"}


agent = QAgent(observation_space, action_space)

nr_winner = {WHITE: [], BLACK: []}

last_state = None

rews = []

def run_game(env):
    # Must create multiple envs, pass in one env per thread
    obs, starting_agent = env.reset()
    winner, done = None, False


    m_rounds = 0

    while not done:

        current_agent = env.current_agent

        if env.current_agent == starting_agent:
            obs, done, winner, rew = agent.apply_action(env, False, m_rounds)
        else:
            obs, done, winner, rew = agent.apply_action(env, True, m_rounds)

        env.change_player_turn()
        m_rounds += 1

        if m_rounds > 95:
            print(COLORS[env.current_agent])
            print(env.get_valid_actions())
            env.render()
        if m_rounds == 100:
            return -1


    return winner


def run_multiple_games(n_threads):

    # Create a environment for each thread

    envs = [gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0') for i in range(n_threads)]

    # Use threadpoolexecutor for easy management of threads
    results = []
    pool = mp.Pool(processes=n_threads)
    for i in tqdm(enumerate(range(EPISODES))):
        result = [pool.map(run_game, envs)]
        results.extend(result[0])
    print(results)
    return results

tic = time.perf_counter()
result = []
env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')
for i in tqdm(enumerate(range(EPISODES))):
    result.append(run_game(env))
toc = time.perf_counter()

print("Time:", round(toc - tic, 2))

save(f"q_tables/q_table_{EPISODES}.npy", agent.Q)

for idx, i in enumerate(result):
    if i == 0:
        nr_winner[0].append(1)
        nr_winner[1].append(0)
    elif i == -1:
        print("Game", idx, "did not finish")
    else:
        nr_winner[0].append(0)
        nr_winner[1].append(1)

wh = sum(nr_winner[0])
bl = sum(nr_winner[1])

print("WHITE:", wh)
print("BLACK:", bl)
print("RATIO:", bl / (wh + bl))