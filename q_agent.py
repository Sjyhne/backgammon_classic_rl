import gym
import numpy as np
import random
from tqdm import tqdm
import time
import multiprocessing as mp
import math

from numpy import save

import matplotlib.pyplot as plt

#[0, 1, 2, 3, 4]
# O, -, XX, -, -
# [2, 2, 2, 2]


EPISODES = 60_000 
STEPS = 2000

class RandomAgent:
    def __init__(self):
        ...

    def apply_random_action(self, environment, observation):
        num_actions = environment.get_n_actions()
        executed = False
        obs = observation

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

class QAgent:
    
    def __init__(self, obs_space, action_space, lr=0.0001, discount=0.95, epsilon=1):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon

        self.actions_executed = np.zeros((8, 8), np.int16)

        self.rewards = []
        # Two last actions
        self.last_action = None
        self.last_obsvervations = None

        self.knockouts = 0

        self.epsilons = []
        
        # TODO: Implement
        #self.start_epsilon_decay = 1
        #self.end_epsilon_decay = EPISODES * 0.9
        #self.epsilon_decay_value = self.epsilon/(self.end_epsilon_decay - self.start_epsilon_decay)

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

    def update_last_actions(self, last_action, last_observation, last_next_observation):
        self.last_action = last_action

        self.last_observation = [last_observation, last_next_observation]

    def decay_epsilon(self, m_rounds, tot_rounds):
        self.epsilons.append(self.epsilon)
        self.epsilon = math.cos(np.linspace(0, 6.28/4, tot_rounds)[m_rounds - 1])

    def get_new_q_value(self, obs, next_obs, reward, action):
        # The best possible future Q value
        max_future_q = np.max(self.Q[next_obs])
        # The Q value for the current action
        current_q = self.Q[obs + action]
        # The new Q for this action in the given observation state
        new_q = current_q + self.lr * (reward + self.discount * max_future_q - current_q)
        # return the new Q value
        return new_q

    def apply_action(self, environment, observation, m_rounds):
        num_actions = environment.get_n_actions()
        executed = False
        obs = observation

        rew = 0

        #print("ROLL:", environment.gym.non_used_dice)

        for _ in range(num_actions):
            actions = [i[1] for i in environment.get_actions()]

            #print(environment.get_valid_actions())
            c = 0

            for _ in range(len(actions)):

                if environment.gym.off[environment.current_agent] == environment.gym.n_pieces:
                    break
                    
                if random.uniform(0, 1) < self.epsilon:
                    action = self.get_random_action_given_observation(actions)
                else:
                    action = self.get_best_action_given_observation(obs)
                    #print("BEST:", action)
                
                next_observation, reward, done, winner, executed = environment.step(action)

                rew += reward
                
                new_q_value = self.get_new_q_value(obs, next_observation, reward, action)

                self.update_Q(obs, action, new_q_value)

                if executed:
                    self.update_last_actions(action, obs, next_observation)
                    obs = next_observation
                    self.actions_executed[action] += 1
                    #print("Q EXECUTED:", action)
                    break
                else:
                    if action in actions:
                        actions.remove(action)

                    c += 1

            if c == 64:
                break


        return obs, done, winner, rew
            
            


#env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

observation_space = (9, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2)
action_space = (8, 8) # 0, ..., 7

WHITE = 0
BLACK = 1

COLORS = {WHITE: "White", BLACK: "Black"}


agents = {WHITE: RandomAgent(), BLACK: QAgent(observation_space, action_space)}

nr_winner = {WHITE: [], BLACK: []}

last_state = None

game_rewards = []

def run_game(env, episode):
    # Must create multiple envs, pass in one env per thread
    obs, current_agent = env.reset()
    winner, done = None, False

    m_rounds = 0

    rews = 0

    while not done:

        agent = agents[env.current_agent]

        rew = 0

        if env.current_agent == 1:
            obs, done, winner, rew = agents[1].apply_action(env, obs, m_rounds)
        else:
            obs, done, winner = agents[0].apply_random_action(env, obs)

        rews += rew

        # if winner != None:
        #     nr_winner[winner].append(1)
        #     nr_winner[env.gym.get_opponent_color(winner)].append(0)

        env.change_player_turn()

        m_rounds += 1

        if m_rounds > 195:
            print(COLORS[env.current_agent])
            print(env.get_valid_actions())
            env.render()
        if m_rounds == 200:
            env.render()
            return -1, rews, m_rounds

        if done:
            if winner == WHITE:
                new_q = agents[BLACK].get_new_q_value(agents[1].last_observation[0], agents[1].last_observation[1], -1, agents[1].last_action)
                agents[BLACK].update_Q(agents[BLACK].last_observation[0], agents[BLACK].last_action, new_q)
                agents[BLACK].decay_epsilon(episode, EPISODES)
                return winner, rews, m_rounds
            elif winner == BLACK:
                new_q = agents[BLACK].get_new_q_value(agents[1].last_observation[0], agents[1].last_observation[1], 1, agents[1].last_action)
                agents[BLACK].update_Q(agents[BLACK].last_observation[0], agents[BLACK].last_action, new_q)
                agents[BLACK].decay_epsilon(episode, EPISODES)
                return winner, rews, m_rounds
            else:
                new_q = agents[BLACK].get_new_q_value(agents[1].last_observation[0], agents[1].last_observation[1], -1, agents[1].last_action)
                agents[BLACK].update_Q(agents[BLACK].last_observation[0], agents[BLACK].last_action, new_q)
                agents[BLACK].decay_epsilon(episode, EPISODES)
                return winner, rews, m_rounds

        
        #print("PLAYER TURN:", COLORS[env.current_agent])
    #print()
    #print(f"Winner: {winner}, Rewards: {rews}, Rounds: {m_rounds}")

    return winner, rews, m_rounds


tic = time.perf_counter()
result = []
rounds = []
knockouts = []
env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')
for _, i in tqdm(enumerate(range(EPISODES))):
    winner, game_reward, m_rounds = run_game(env, i + 1)
    result.append(winner)
    game_rewards.append(game_reward)    
    rounds.append(m_rounds)
    knockouts.append(agents[BLACK].knockouts)
    agents[BLACK].knockouts = 0
    if i % STEPS == 0:
        # print(agents[1].Q[2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 0])
        # print(agents[1].Q[2, 0, 6, 0, 2, 5, 5, 0, 0, 1, 0])
        try:
            print("WIN RATIO: ------------------------------------ ", sum(result[-STEPS:])/STEPS)
        except:
            print("WOPS")
toc = time.perf_counter()

save(f"q_tables/q_table_{EPISODES}_{STEPS}_.npy", agents[1].Q)

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

avgs = [sum(game_rewards[x:x + STEPS])/STEPS for x in range(0, len(game_rewards), STEPS)]
wins = [sum(nr_winner[BLACK][x:x + STEPS])/STEPS for x in range(0, len(nr_winner[BLACK]), STEPS)]
rounds = [sum(rounds[x:x + STEPS])/STEPS for x in range(0, len(rounds), STEPS)]
knockouts = [sum(knockouts[x:x + STEPS])/STEPS for x in range(0, len(knockouts), STEPS)]


plt.plot(avgs)
plt.title(f"AVG REWARDS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(wins)
plt.title(f"WINS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(wins)
plt.title(f"AVG ROUNDS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(agents[1].epsilons)
plt.title(f"EPSILONS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(knockouts)
plt.title(f"KNOCKOUTS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.imshow(agents[BLACK].actions_executed, cmap="hot", interpolation="nearest")
plt.show()

print(len(agents[1].epsilons))

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