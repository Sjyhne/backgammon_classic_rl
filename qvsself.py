import gym
import numpy as np
import random
from tqdm import tqdm
import time
import multiprocessing as mp

from numpy import load

import math

from utils import flip_observation, flip_action

import matplotlib.pyplot as plt

# save numpy array as npy file
from numpy import asarray
from numpy import save, savetxt


EPISODES = 2
STEPS = 1

class QAgent:
    
    def __init__(self, obs_space, action_space, lr=0.0001, discount=0.95, epsilon=0, q_path=None):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon

        self.actions_executed = np.zeros((8, 8), np.int16)

        self.did_not_execute = 0

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

        self.Q = self.initiate_Q(obs_space, action_space, q_path=q_path)
    
    
    def initiate_Q(self, obs_space, action_space, q_path=None):
        # Initiates the Q table using the obs space and the action space
        if q_path != None:
            return load(q_path)
        else:
            return np.zeros((obs_space + action_space), np.float16)

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
        self.epsilon = math.cos(np.linspace(6.28/4.2, 6.28/4, tot_rounds)[m_rounds - 1])

    def get_new_q_value(self, obs, next_obs, reward, action):
        # The best possible future Q value
        max_future_q = np.max(self.Q[next_obs])
        # The Q value for the current action
        current_q = self.Q[obs + action]
        # The new Q for this action in the given observation state
        new_q = current_q + self.lr * (reward + self.discount * max_future_q - current_q)
        # return the new Q value
        return new_q

    def apply_action(self, environment, m_rounds, flipped):
        num_actions = environment.get_n_actions()
        executed = False
        obs = tuple(environment.gym.get_current_observation())

        if flipped:
            obs = flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots)

        rew = 0

        #print("ROLL:", environment.gym.non_used_dice)

        for _ in range(num_actions):

            if environment.gym.off[environment.current_agent] == environment.gym.n_pieces:
                break
            
            if random.uniform(0, 1) < self.epsilon:
                actions = []
                if flipped:
                    actions = [flip_action(i[1], environment.gym.n_spots) for i in environment.get_actions()]
                else:
                    actions = [i[1]for i in environment.get_actions()]

                c = 0

                for _ in range(len(actions)):
                    action = self.get_random_action_given_observation(actions)

                    next_observation, reward, done, winner, executed = environment.step(action)

                    rew += reward
                    if flipped:
                        new_q_value = self.get_new_q_value(flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), 
                            flip_observation(next_observation, environment.gym.n_pieces, environment.gym.n_spots), reward, 
                            flip_action(action, environment.gym.n_spots))

                        self.update_Q(flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), flip_action(action, environment.gym.n_spots), new_q_value)
                    else:
                        new_q_value = self.get_new_q_value(obs, next_observation, reward, action)

                        self.update_Q(obs, action, new_q_value)

                    if executed:
                        self.update_last_actions(flip_action(action, environment.gym.n_spots), flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), next_observation)
                        obs = next_observation
                        if flipped:
                            self.actions_executed[flip_action(action, environment.gym.n_spots)] += 1
                        else:
                            self.actions_executed[action] += 1
                        print("R EXECUTED:", action)
                        break
                    else:
                        if action in actions:
                            actions.remove(action)
                        
                        c += 1

                if c == 64:
                    break    
                
            else:
                action = self.get_best_action_given_observation(obs)

                next_observation, reward, done, winner, executed = environment.step(action)

                if flipped:
                    new_q_value = self.get_new_q_value(flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), flip_observation(next_observation, environment.gym.n_pieces, environment.gym.n_spots), reward, flip_action(action, environment.gym.n_spots))
                    self.update_Q(flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), flip_action(action, environment.gym.n_spots), new_q_value)
                else:
                    new_q_value = self.get_new_q_value(obs, next_observation, reward, action)

                    self.update_Q(obs, action, new_q_value)

                if executed:
                    self.update_last_actions(flip_action(action, environment.gym.n_spots), flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), next_observation)
                    obs = next_observation
                    if flipped:
                        self.actions_executed[flip_action(action, environment.gym.n_spots)] += 1
                    else:
                        self.actions_executed[action] += 1
                    print("Q EXECUTED:", action)
                    break
                else:
                    # Because get best action always will choose the same action, then I had to
                    # Make a backup where it just executes a random action if the best action 
                    # Is an action that cannot be executed.
                    self.did_not_execute += 1
                    actions = []
                    if flipped:
                        actions = [flip_action(i[1], environment.gym.n_spots) for i in environment.get_actions()]
                    else:
                        actions = [i[1]for i in environment.get_actions()]

                    c = 0

                    for _ in range(len(actions)):
                        action = self.get_random_action_given_observation(actions)

                        next_observation, reward, done, winner, executed = environment.step(action)

                        rew += reward
                        if flipped:
                            new_q_value = self.get_new_q_value(flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), 
                                flip_observation(next_observation, environment.gym.n_pieces, environment.gym.n_spots), reward, 
                                flip_action(action, environment.gym.n_spots))

                            self.update_Q(flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), flip_action(action, environment.gym.n_spots), new_q_value)
                        else:
                            new_q_value = self.get_new_q_value(obs, next_observation, reward, action)

                            self.update_Q(obs, action, new_q_value)

                        if executed:
                            self.update_last_actions(flip_action(action, environment.gym.n_spots), flip_observation(obs, environment.gym.n_pieces, environment.gym.n_spots), next_observation)
                            obs = next_observation
                            if flipped:
                                self.actions_executed[flip_action(action, environment.gym.n_spots)] += 1
                            else:
                                self.actions_executed[action] += 1
                            print("R EXECUTED:", action)
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

try:
    black_agent = QAgent(observation_space, action_space)
except Exception as e:
    print("BLACK:", e)

try:
    white_agent = QAgent(observation_space, action_space, q_path="q_tables/q_table_10000_1000_white.npy")
except Exception as e:
    print("WHITE:", e)

agents = {WHITE: white_agent, BLACK: white_agent}

nr_winner = {WHITE: [], BLACK: []}

last_state = None

game_rewards = {WHITE: [], BLACK: []}

didnt_execute = []

starting_color = {WHITE: 0, BLACK: 0}

# Agent should see the board as "black"

def run_game(env, episode):

    obs, current_agent = env.reset()
    starting_color[current_agent] += 1
    winner, done = None, False

    m_rounds = 0

    rews = {WHITE: 0, BLACK: 0}
    
    print("EPS:", agents[WHITE].epsilon)

    env.render()

    while not done:

        agent = agents[env.current_agent]


        rew = 0

        if env.current_agent == 1:
            obs, done, winner, rew = agent.apply_action(env, m_rounds, True)
            rews[BLACK] += rew
        else:
            obs, done, winner, rew = agent.apply_action(env, m_rounds, False)
            rews[WHITE] += rew

        
        env.change_player_turn()
        print("Current agent:", COLORS[env.current_agent])
        env.render()

        m_rounds += 1

        if m_rounds > 195:
            print(COLORS[env.current_agent])
            print(env.get_valid_actions())
            env.render()
        if m_rounds == 200:
            env.render()
            return -1, rews, m_rounds

        if done:
            didnt_execute.append(agent.did_not_execute)
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
knockouts = {WHITE: [], BLACK: []}
qs = {WHITE: [], BLACK: []}
env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')
for _, i in tqdm(enumerate(range(EPISODES))):
    winner, game_reward, m_rounds = run_game(env, i + 1)
    result.append(winner)
    game_rewards[WHITE].append(game_reward[WHITE])
    game_rewards[BLACK].append(game_reward[BLACK])    
    rounds.append(m_rounds)
    knockouts[BLACK].append(agents[BLACK].knockouts)
    knockouts[WHITE].append(agents[WHITE].knockouts)
    #if (i % STEPS == 0 and i != 0) or i == EPISODES - 1:
        #print("Kill 1")
        #qs[WHITE].append(np.absolute(agents[WHITE].Q).sum())
        #print("Kill 2")
        #qs[BLACK].append(np.absolute(agents[BLACK].Q).sum())
    if i % STEPS == 0 and i != 0:
        #print(agents[1].Q[2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 1])
        #print(agents[1].Q[2, 0, 6, 0, 2, 5, 5, 0, 0, 0, 1])
        #print(agents[1].Q[2, 0, 6, 5, 2, 0, 5, 0, 0, 1, 1])
        try:
            print()
            print("WIN BLACK RATIO: ------------------------------------ ", sum(result[-STEPS:])/STEPS)
            print("WIN WHITE RATIO: ------------------------------------ ", 1 - sum(result[-STEPS:])/STEPS)
        except:
            print("WOPS")
toc = time.perf_counter()

#save(f"q_tables/q_table_{EPISODES}_{STEPS}_white.npy", agents[1].Q)
#save(f"q_tables/q_table_{EPISODES}_{STEPS}_white.npy", agents[0].Q)


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

print("Starting color: WHITE:", starting_color[WHITE] / (starting_color[WHITE] + starting_color[BLACK]), "BLACK:", starting_color[BLACK] / (starting_color[WHITE] + starting_color[BLACK]))

black_avgs = [sum(game_rewards[BLACK][x:x + STEPS])/STEPS for x in range(0, len(game_rewards[BLACK]), STEPS)]
white_avgs = [sum(game_rewards[WHITE][x:x + STEPS])/STEPS for x in range(0, len(game_rewards[WHITE]), STEPS)]
black_wins = [sum(nr_winner[BLACK][x:x + STEPS])/STEPS for x in range(0, len(nr_winner[BLACK]), STEPS)]
white_wins = [sum(nr_winner[WHITE][x:x + STEPS])/STEPS for x in range(0, len(nr_winner[WHITE]), STEPS)]
rounds = [sum(rounds[x:x + STEPS])/STEPS for x in range(0, len(rounds), STEPS)]



plt.plot(black_avgs, label="BLACK")
plt.plot(white_avgs, label="WHITE")
plt.legend()
plt.title(f"AVG REWARDS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(black_wins, label="BLACK")
plt.plot(white_wins, label="WHITE")
plt.legend()
plt.title(f"WINS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(qs[BLACK], label="BLACK")
plt.plot(qs[WHITE], label="WHITE")
plt.legend()
plt.title(f"abs(Q).sum() | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(rounds)
plt.title(f"AVG ROUNDS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.plot(agents[1].epsilons)
plt.title(f"EPSILONS | EP: {EPISODES}, lr:{agents[BLACK].lr}, eps: {agents[BLACK].epsilon}, stps: {STEPS}, disc: {agents[BLACK].discount}")
plt.show()

plt.imshow(agents[BLACK].actions_executed, cmap="hot", interpolation="nearest")
plt.show()

print(len(agents[1].epsilons))