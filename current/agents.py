import numpy as np
import random
import math
from numpy import load

from utils import flip_action, flip_observation


class QAgent():

    # Train must be false when measuring the accuracy of the model - If not it will be using random actions
    # To make a play. This is included in training because it increases the training experience for each
    # Observation met when the best q value is not a valid play.
    def __init__(self, obs_space, action_space, lr=0.0001, discount=0.9, epsilon=1, train=True, load_path=None, print=False):
        
        # Agent configuration
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.load_path = load_path
        self.train = train
        self.print = print

        # Agent Q Table
        self.Q = self.initiate_Q(obs_space, action_space, load_path)

        # Last action and observations
        self.last_action = ()
        self.last_observations = []

        self.random_actions = 0
        self.Q_actions = 0

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
        self.last_observations = [last_observation, last_next_observation]
    
    def get_best_action(self, obs):
        action = np.unravel_index(np.argmax(self.Q[obs], axis=None), self.Q[obs].shape)
        return action

    def get_random_action(self, actions):
        return random.choice(actions)

    def decay_epsilon(self, episode, episodes):
        
        # Logging
        self.epsilons.append(self.epsilon)

        if self.epsilon != 0:
            # Update epsilon following a ^2 curve
            if math.cos(np.linspace((6.7/4) * (1 - self.epsilon), 6.7/4, episodes)[episode]) > 0:
                self.epsilon = math.cos(np.linspace(0, 6.7/4, episodes)[episode])
            else:
                self.epsilon = 0

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
        new_q = self.calculate_new_q(obs, next_obs, action, reward)
        self.update_Q(obs, action, new_q)
        # If the action wasnt executed then continue to loop through the actions
        if not executed:
            if action in actions:
                actions.remove(action)
            return False
        else:
            # Update the last action made
            self.update_last_actions(action, obs, next_obs)
            self.actions_executed[action] += 1
            return True

    def execute_action(self, env):

        done = False
        winner = None

        num_actions = env.get_n_actions()

        for i in range(num_actions):

            random_actions = [i[1] for i in env.get_actions()]

            obs = env.get_current_observation()

            # Check if the agent should to random or best action
            if random.uniform(0, 1) < self.epsilon:

                # Number of random actions
                n_actions = len(random_actions)

                for n in range(n_actions):
                    # Choose a random action
                    random_action = self.get_random_action(random_actions)


                    # Try to perform the random action
                    next_obs, reward, done, winner, executed = env.step(random_action)

                    if executed:
                        self.random_actions += 1
                    
                    if self.step(obs, next_obs, reward, done, winner, executed, random_action, random_actions):
                        break
                    else:
                        continue
            else:
                
                action = self.get_best_action(obs)
                next_obs, reward, done, winner, executed = env.step(action)

                if executed:
                    self.Q_actions += 1
                
                if not executed and self.print:
                    if len(env.get_valid_actions()) != 0:
                        print("Failed to execute:", action, "Possible:", env.get_valid_actions(), "Dice:", env.gym.non_used_dice)
                        env.render()
                        print(self.Q[obs])

                # This is implemented in order to learn the agent faster than if it should've
                # just tried the "best" action from the current q table. I think. This might
                # Be good report meat
                if not executed and self.train:
                    # Number of random actions
                    n_actions = len(random_actions)

                    for n in range(n_actions):
                        # Choose a random action
                        random_action = self.get_random_action(random_actions)


                        # Try to perform the random action
                        next_obs, reward, done, winner, executed = env.step(random_action)
                        
                        if executed:
                            self.random_actions += 1

                        if self.step(obs, next_obs, reward, done, winner, executed, random_action, random_actions):
                            break
                        else:
                            continue
                else:
                    self.step(obs, next_obs, reward, done, winner, executed, action, random_actions)
        
        return obs, done, winner, sum(self.rewards)


class QAgentFlip():

    # Train must be false when measuring the accuracy of the model - If not it will be using random actions
    # To make a play. This is included in training because it increases the training experience for each
    # Observation met when the best q value is not a valid play.
    def __init__(self, obs_space, action_space, lr=0.0001, discount=0.95, epsilon=1, train=True, load_path=None):
        
        # Agent configuration
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.load_path = load_path
        self.train = train

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
        self.last_observations = [last_observation, last_next_observation]
    
    def get_best_action(self, obs):
        action = np.unravel_index(np.argmax(self.Q[obs], axis=None), self.Q[obs].shape)
        return action

    def get_random_action(self, actions):
        return random.choice(actions)

    def decay_epsilon(self, episode, episodes):
        
        # Logging
        self.epsilons.append(self.epsilon)

        if self.epsilon != 0:
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
        new_q = self.calculate_new_q(obs, next_obs, action, reward)
        self.update_Q(obs, action, new_q)

        # If the action wasnt executed then continue to loop through the actions
        if not executed:
            if action in actions:
                actions.remove(action)
            return False
        else:
            # Update the last action made
            self.update_last_actions(action, obs, next_obs)
            self.actions_executed[action] += 1
            return True

    def execute_action(self, env, flip=False):

        done = False
        winner = None

        num_actions = env.get_n_actions()

        for i in range(num_actions):
            
            # There are only flipped actions in random actions now - Therefore must be flipped
            # Before execution

            if flip:
                random_actions = [flip_action(i[1], env.gym.n_spots) for i in env.get_actions()]
            else:
                random_actions = [i[1] for i in env.get_actions()]

            obs = env.get_current_observation()

            # Flip observation
            if flip:
                obs = flip_observation(obs, env.gym.n_pieces, env.gym.n_spots)

            # Check if the agent should to random or best action
            if random.uniform(0, 1) < self.epsilon:

                # Number of random actions
                n_actions = len(random_actions)

                for n in range(n_actions):
                    # Choose a random action
                    random_action = self.get_random_action(random_actions)


                    # Try to perform the random action | Flip action before executing
                    if flip:
                        next_obs, reward, done, winner, executed = env.step(flip_action(random_action, env.gym.n_spots))
                    else:
                        next_obs, reward, done, winner, executed = env.step(random_action)

                    if flip:
                        next_obs = flip_observation(next_obs, env.gym.n_pieces, env.gym.n_spots)
                    
                    if self.step(obs, next_obs, reward, done, winner, executed, random_action, random_actions):
                        break
                    else:
                        continue
            else:

                action = self.get_best_action(obs)

                if flip:
                    next_obs, reward, done, winner, executed = env.step(flip_action(action, env.gym.n_spots))
                else:
                    next_obs, reward, done, winner, executed = env.step(action)

                if flip:
                    next_obs = flip_observation(next_obs, env.gym.n_pieces, env.gym.n_spots)

                # This is implemented in order to learn the agent faster than if it should've
                # just tried the "best" action from the current q table. I think. This might
                # Be good report meat
                if not executed and self.train:
                    # Number of random actions
                    n_actions = len(random_actions)

                    for n in range(n_actions):
                        # Choose a random action
                        random_action = self.get_random_action(random_actions)


                        # Try to perform the random action
                        if flip:
                            next_obs, reward, done, winner, executed = env.step(flip_action(random_action, env.gym.n_spots))
                        else:
                            next_obs, reward, done, winner, executed = env.step(random_action)

                        if flip:
                            next_obs = flip_observation(next_obs, env.gym.n_pieces, env.gym.n_spots)

                        
                        if self.step(obs, next_obs, reward, done, winner, executed, random_action, random_actions):
                            break
                        else:
                            continue
                else:
                    self.step(obs, next_obs, reward, done, winner, executed, action, random_actions)
        
        return obs, done, winner, sum(self.rewards)



class RandomAgent:
    def __init__(self):
        ...

    def apply_random_action(self, environment):
        num_actions = environment.get_n_actions()
        executed = False
        obs = environment.get_current_observation()

        done = False
        winner = None

        #print("ROLL:", environment.gym.non_used_dice)

        for _ in range(num_actions):
            actions = environment.gym.get_valid_actions(environment.current_agent)
            acts = [i[1] for i in actions]
            #print(environment.get_valid_actions())
            c = 0
            if len(actions) > 0:
                for _ in actions:
                    action = random.choice(acts)
                    next_observation, reward, done, winner, executed = environment.step(action)
                    if executed:
                        obs = next_observation
                        #print("R EXECUTED:", action)
                        break
                    else:
                        acts.remove(action)
                        c += 1

                if c == len(acts):
                    break
        
        return obs, done, winner