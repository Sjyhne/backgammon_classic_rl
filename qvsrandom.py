import gym
import numpy as np
import random
from tqdm import tqdm
import time
import multiprocessing as mp

import matplotlib.pyplot as plt

from numpy import load


#[0, 1, 2, 3, 4]
# O, -, XX, -, -
# [2, 2, 2, 2]


EPISODES = 1000
STEPS = 20

class RandomAgent:
    def __init__(self):
        ...

    def apply_random_action(self, environment):
        num_actions = environment.get_n_actions()
        executed = False
        obs = environment.gym.get_current_observation(environment.current_agent)

        for _ in range(num_actions):
            actions = environment.get_actions()
            acts = [i[1] for i in actions]
            #print(environment.get_valid_actions())
            c = 0

            for _ in actions:
                action = random.choice(acts)
                next_observation, reward, done, winner, executed = environment.step(action)
                if executed:
                    obs = next_observation
                    #print("EXECUTED:", action)
                    break
                else:
                    acts.remove(action)
                    c += 1

            if c == len(acts):
                break
        
        return obs, done, winner

class QAgent:
    
    def __init__(self):

        self.Q = load('q_tables/q_table_50.npy')
    
    
    # def initiate_Q(self, obs_space, action_space):
    #     # Initiates the Q table using the obs space and the action space
    #     return(np.zeros((obs_space + action_space), np.float16))

    # def update_Q(self, obs, action, value):
    #     # Updates the Q table
    #     #print(self.Q[obs + action])
    #     self.Q[(obs + action)] = value

    def get_best_action_given_observation(self, obs):
        # Get the best action for the given observation (0, 2)
        action = np.unravel_index(np.argmax(self.Q[obs], axis=None), self.Q[obs].shape)

        return action

    # def get_random_action_given_observation(self, actions):
    #     action = random.choice(actions)
    #     return action

    # def decay_epsilon(self):
    #     self.epsilon -= self.epsilon_decay_value

    # def get_new_q_value(self, obs, next_obs, reward, action):
    #     # The best possible future Q value
    #     max_future_q = np.max(self.Q[next_obs])
    #     # The Q value for the current action
    #     current_q = self.Q[obs + action]
    #     # The new Q for this action in the given observation state
    #     new_q = (1 - self.lr) * current_q + self.lr * (reward + self.discount * max_future_q)
    #     # return the new Q value
    #     return new_q

    def apply_action(self, environment):
        num_actions = environment.get_n_actions()
        executed = False
        obs = environment.gym.get_current_observation(environment.current_agent)
        for _ in range(num_actions):
            actions = [i[1] for i in environment.get_actions()]

            l_acts = len(actions)

            c = 0

            for _ in range(len(actions)):


                action = self.get_best_action_given_observation(obs)
                #print("BEST:", action)
                
                next_observation, reward, done, winner, executed = environment.step(action)

                if executed:
                    obs = next_observation
                    break
                else:
                    if action in actions:
                        actions.remove(action)
                    c += 1

            if c == l_acts:
                break

        return obs, done, winner
            
            

WHITE = 0
BLACK = 1

COLORS = {WHITE: "White", BLACK: "Black"}


agents = {WHITE: RandomAgent(), BLACK: QAgent()}

nr_winner = {WHITE: [], BLACK: []}

last_state = None

rews = []

def run_game(env):
    # Must create multiple envs, pass in one env per thread
    obs, current_agent = env.reset()
    winner, done = None, False

    m_rounds = 0

    while not done:

        agent = agents[env.current_agent]

        if env.current_agent == 1:
            obs, done, winner, rew = agents[1].apply_action(env)
        else:
            obs, done, winner = agents[0].apply_random_action(env)

        env.change_player_turn()
        m_rounds += 1

        if m_rounds > 95:
            env.render()
        if m_rounds == 100:
            env.render()
            return -1

    return winner

result = []

tic = time.perf_counter()
env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')
for i in tqdm(enumerate(range(EPISODES))):
    result.append(run_game(env))
toc = time.perf_counter()

print("Time:", round(toc - tic, 2))

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



"""if i % 100 == 0:
    print()
    print("BLACK:", sum(nr_winner[BLACK]) / (sum(nr_winner[BLACK]) + sum(nr_winner[WHITE])))
    print("WHITE:", sum(nr_winner[WHITE]) / (sum(nr_winner[BLACK]) + sum(nr_winner[WHITE])))"""

"""for _, i in tqdm(enumerate(range(EPISODES))):

    obs, current_agent = env.reset()
    winner, done = None, False

    tot_rew = 0

    while not done:

        agent = agents[env.current_agent]
        last_state = obs


        #print(agent)
        #print(env.gym.non_used_dice)

        if i == EPISODES - 1:
            env.render()


        if env.current_agent == BLACK:
            obs, done, winner, rew = agent.apply_action(env, obs)
            tot_rew += rew
        else:
            obs, done, winner = agent.apply_random_action(env, obs)

        if winner != None:
            #print("WINNER IS:", COLORS[winner])
            nr_winner[winner].append(1)
            nr_winner[env.gym.get_opponent_color(winner)].append(0)
        
        env.change_player_turn()


        #env.render()
    
    rews.append(tot_rew)

    if i % 100 == 0:
        print()
        print("BLACK:", sum(nr_winner[BLACK]) / (sum(nr_winner[BLACK]) + sum(nr_winner[WHITE])))
        print("WHITE:", sum(nr_winner[WHITE]) / (sum(nr_winner[BLACK]) + sum(nr_winner[WHITE])))

print(agents[BLACK].Q[last_state])
print(agents[BLACK].get_best_action_given_observation(last_state))
print(agents[BLACK].Q[last_state + agents[BLACK].get_best_action_given_observation(last_state)])
print(last_state)

print(sum(nr_winner[BLACK][int(EPISODES * 0.9):]))
print(sum(nr_winner[WHITE][int(EPISODES * 0.9):]))

avgs = [sum(rews[x:x + STEPS])/STEPS for x in range(0, len(rews), STEPS)]
wins = [sum(nr_winner[BLACK][x:x + STEPS])/STEPS for x in range(0, len(nr_winner[BLACK]), STEPS)]


plt.plot(avgs)
plt.title(f"AVG REWARDS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(wins)
plt.title(f"WINS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()"""