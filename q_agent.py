import gym
import numpy as np
import random
from tqdm import tqdm

#[0, 1, 2, 3, 4]
# O, -, XX, -, -
# [2, 2, 2, 2]

class RandomAgent:
    def __init__(self):
        ...

    def apply_random_action(self, environment, observation):
        num_actions = environment.get_n_actions()
        executed = False
        obs = observation

        for _ in range(num_actions):
            actions = environment.get_actions()
            acts = [i[1] for i in actions]
            #print(environment.get_valid_actions())
            c = 0

            for _ in actions:
                action = random.choice(acts)
                next_observation, reward, done, winner, executed = env.step(action)
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
    
    def __init__(self, obs_space, action_space, lr=0.0001, discount=0.9, epsilon=0.6):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        
        # TODO: Implement
        self.start_epsilon_decay = 1
        self.end_epsilon_decay = 10_000//2
        self.epsilon_decay_value = self.epsilon/(self.end_epsilon_decay - self.start_epsilon_decay)

        self.Q = self.initiate_Q(obs_space, action_space)
    
    
    def initiate_Q(self, obs_space, action_space):
        # Initiates the Q table using the obs space and the action space
        return(np.zeros((obs_space + action_space), np.float16))

    def update_Q(self, obs, action, value):
        # Updates the Q table
        #print(self.Q[obs + action])
        self.Q[(obs + action)] = value

    def get_best_action_given_observation(self, obs):
        # Get the best action for the given observation (0, 2)
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

    def apply_action(self, environment, observation):
        num_actions = environment.get_n_actions()
        executed = False
        obs = observation

        for _ in range(num_actions):
            actions = [i[1] for i in environment.get_actions()]

            #print(environment.get_valid_actions())
            c = 0

            for _ in range(len(actions)):

                if random.uniform(0, 1) < self.epsilon:
                    action = self.get_random_action_given_observation(actions)
                else:
                    action = self.get_best_action_given_observation(obs)
                    #print("BEST:", action)
                
                next_observation, reward, done, winner, executed = env.step(action)
                
                new_q_value = self.get_new_q_value(obs, next_observation, reward, action)
                self.update_Q(obs, action, new_q_value)

                if executed:
                    #print("EXECUTED:", action)
                    obs = next_observation
                    #environment.render()
                    break
                else:
                    if action in actions:
                        actions.remove(action)
                        #print("REMOVED:", action)
                    c += 1

            if c == 64:
                break
        
        return obs, done, winner
            
            


env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

# By printing the observation_space and action_space we see that
# observation_space = [9, 9, 9, 9, 9, 9, 9, 2, 2, 3, 2] and action_space = [8, 8]
observation_space = (9, 9, 9, 9, 9, 9, 9, 2, 2, 3, 2)
action_space = (8, 8) # 0, ..., 7

# SRC: 0, 1, 2, 3, 4, 5, 6, BAR
# DST: 0, 1, 2, 3, 4, 5, 6, OFF

# 5, 6, OFF
# O, O
# ROLL: (2, 1)
# ACTION: (6, OFF)

"""
agent = QAgent(observation_space, action_space)
agent.initiate_Q(observation_space, action_space)
print(agent.Q.shape)
print("Q table takes up:", round(((agent.Q.size * agent.Q.itemsize) / (1024 ** 2)), 1), "MB RAM")

obs, current_player = tuple(env.reset())

print((obs) + (action_space))

print(agent.Q[tuple(obs)])

for i, row in enumerate(agent.Q[obs]):
    for idx, item in enumerate(row):
        agent.update_Q(obs, (i, idx), idx - i * 0.1)

print(agent.Q[obs])

print(np.unravel_index(np.argmax(agent.Q[obs], axis=None), agent.Q[obs].shape))
"""
WHITE = 0
BLACK = 1

COLORS = {WHITE: "White", BLACK: "Black"}


agents = {WHITE: RandomAgent(), BLACK: QAgent(observation_space, action_space)}

nr_winner = {WHITE: 0, BLACK: 0}

last_state = None

for _, i in tqdm(enumerate(range(10_000))):

    obs, current_agent = env.reset()
    winner, done = None, False

    #env.render()

    while not done:

        agent = agents[env.current_agent]
        last_state = obs


        #print(agent)
        #print(env.gym.non_used_dice)

        if env.current_agent == BLACK:
            obs, done, winner = agent.apply_action(env, obs)
        else:
            obs, done, winner = agent.apply_random_action(env, obs)

        if winner != None:
            #print("WINNER IS:", COLORS[winner])
            nr_winner[winner] += 1
        
        env.change_player_turn()

        #env.render()

    if i % 1000 == 0:
        print(nr_winner)
        print("BLACK:", nr_winner[BLACK] / (nr_winner[BLACK] + nr_winner[WHITE]))
        print("WHITE:", nr_winner[WHITE] / (nr_winner[BLACK] + nr_winner[WHITE]))

print(nr_winner)
print(agents[BLACK].Q[last_state])
print(agents[BLACK].get_best_action_given_observation(last_state))
print(last_state)