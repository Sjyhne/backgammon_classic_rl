import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
import gym
from torch import tensor
from torch.distributions import Categorical
from torch.optim import Adam
from network import feedforwardNN

from agents import RandomAgent

import time



class PPO:
    def __init__(self, env, **hyperparameters): #Hyperparmeters uses idom to allow different changes to only specified paramters to be changed
        
        #Initialize hyperparamters
        self._init_hyperparamters(hyperparameters)

        #Get environment information
        self.env = env
        self.obs_dim = (9, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2)
        #self.obs_dim = env.observation_space.shape[0]
        self.act_dim = (8, 8)
        #self.act_dim = env.action_space

        #Initalize the actor and the critic
        self.actor = feedforwardNN(self.obs_dim, self.act_dim)
        self.critic = feedforwardNN(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)


    #Learning loop
    def learn(self, ep_total):
        ep_iterated = 0 #Episodes iterated so far

        total_wins = [] #Total wins of the learning loop, represneted by 1's and 0's
        total_rewards = []

        while ep_iterated < ep_total:

            print("Percent done:", round(ep_iterated/ep_total*100, 2), "%")

            batch_obs, batch_actions, batch_log_probs, batch_qvals, batch_lens, batch_wins = self.rollout()

            batch_win_percentage = sum(batch_wins)/len(batch_wins)

            total_wins.append(batch_win_percentage)

            print("Batch win acc:", batch_win_percentage)

            # Calculate how many timesteps we collected this batch   
            ep_iterated += np.sum(batch_lens)

            #Calculate V
            V = self.evaluate_values(batch_obs)

            #Calculate advantage at k'th iteration
            A_k = batch_qvals - V.detach()

            #Normalize advantage(helps training stability)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.updates_per_iteration):

                #Calculate current log probabilities
                current_log_probs = self.evaluate_log_probs(batch_obs, batch_actions)

                #Calculate current V values
                V = self.evaluate_values(batch_obs)

                #Calculate ratios. e^ln(x) = x  
                ratios = torch.exp(current_log_probs - batch_log_probs)

                #Calculate surrogate losses
                surrogate1 = ratios * A_k
                surrogate2 = torch.clamp(ratios, 1- self.clip, 1 + self.clip) * A_k #Clamp restrict values to be not higher or smaller than the set min and max, binding values to the respective upper and lower bounds if the values does not adher to the set min and max

                #Calculate actor objective. We use - since we are trying to maximize rewards through sthocastic ascent 
                actor_loss = (-torch.min(surrogate1, surrogate2)).mean()

                #Calcualte critic loss
                critic_loss = nn.MSELoss()(V, batch_qvals)


                #Calculate gradients and perfom backwards propagation on actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph = True)
                self.actor_optim.step()

                 #Calculate gradients and perfom backwards propagation on critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            
            # Save our model if it's time, t_iterated is the amount of learnings steps so far, save_freq is how frequent we should save, so 2010 % 10 = 0 -> Save
            if ep_iterated % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

        return total_wins
        
    #Define hyperparameters
    def _init_hyperparamters(self, hyperparameters):
        #Default values, need to experiment with changes
        self.episode_per_batch = 100            #Timesteps per batch
        self.max_t_per_episode = 1000           #Max timesteps per episode
        self.gamma = 0.95                       #Gamma for discounted return
        self.updates_per_iteration = 5          #Amount of updates per epoch(epoch equals batch size)
        self.clip = 0.2                         #Clip recommended by paper
        self.lr = 0.005                         #Learning rate of optimizers

        self.render = False                     #If we should render during rollout
        self.save_freq = 10                     #How often we save in number of iterations

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))


    def game_loop_vs_random(self):

        episode_obs = []          #Episode observations
        episode_actions = []      #Episode actions  
        episode_log_probs = []    #log probabilities of each action 
        episode_rews = []         #Episode rewards
                
        episode_rews = []         #Rewards this episode
        random_agent = RandomAgent()

        # [0] = obs, [1] = starting agent
        obs, current_agent = self.env.reset()
        done = False
        
        # Run an epsiode for a maximum of max timesteps per episode, or break if terminal state has been reached(game finished)
        for ep_t in range(self.max_t_per_episode):
            
            # Render if it is set as true
            if self.render:
                self.env.render()

            if current_agent == 1:
                
                for _ in range(len(self.env.gym.non_used_dice)):

                    #Collect observation
                    episode_obs.append(obs)

                    action_net, action_env, valid_log_prob = self.get_action(obs)
                    
                    # step returns tuple(current_observation), reward, done, winner, executed
                    obs, rew, done, winner, ex = self.env.step(action_env)

                    episode_rews.append(rew)
                    episode_log_probs.append(valid_log_prob)
                    episode_actions.append(action_net)
            
            else:
                _, done, winner, _ = random_agent.apply_random_action(self.env)

            self.env.change_player_turn()

            current_agent = self.env.current_agent

            if done:
                break
        
        return episode_obs, episode_actions, episode_log_probs, episode_rews, winner

    #Rollout function to collect batch data(need to understand why we use log on porobabilties)
    def rollout(self):
        batch_obs = []          #Batch observations
        batch_actions = []      #Batch actions  
        batch_log_probs = []    #log probabilities of each action 
        batch_rews = []         #Batch rewards
        batch_qvals = []        #Batch q values, index 0 will correspond to the q value at timestep 1 in the first epsiode
        batch_lens = []         #Lengths of episodes in batch
        batch_wins = []         #PPO agents wins and losses represented by 1's and 0's

        episode_iterated = 0 #Timesteps iterated so far in batch

        while episode_iterated < self.episode_per_batch:


            episode_obs, episode_actions, episode_log_probs, episode_rews, winner = self.game_loop_vs_random()

            batch_obs.extend(episode_obs)
            batch_actions.extend(episode_actions)
            batch_log_probs.extend(episode_log_probs)

            if winner != None:
                batch_wins.append(winner)
            
            #Collect episodic length and rewards
            batch_lens.append(1)
            batch_rews.append(episode_rews)

            episode_iterated += 1

    
        #Reshape data as tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.int)
        batch_actions = torch.tensor(batch_actions, dtype=torch.int)
        batch_log_probs = torch.tensor(batch_log_probs, dtype= torch.float)

        #Collect return
        batch_qvals = self.compute_qvals(batch_rews)

        print(sum(batch_wins)/len(batch_wins))

        #Return the batch data
        return batch_obs, batch_actions, batch_log_probs, batch_qvals, batch_lens, batch_wins

    #Q value function, the index is the timestep. at index 1 it has a high value with the discounted future rewards.
    def compute_qvals(self, batch_rews):
        size = 0
        for eps in batch_rews:
            size += len(eps)
        
        #The return per episode per batch   
        batch_qvals = []

        #Iterate in reverse to keep same order in batch  
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_qvals.insert(0, discounted_reward)
        
        #Convert q values into a tensor
        batch_qvals = torch.tensor(batch_qvals, dtype=torch.float)

        #Return the discounted q values
        return batch_qvals
    
    #Use the critic network to evaluate the obeservation Values
    def evaluate_values(self, batch_obs):

        #Query critic network for value V at each observation in batch
        V = self.critic(batch_obs).squeeze() #Squeeze is used to change dimension of the tensor

        return V
    
    def evaluate_log_probs(self, batch_obs, batch_acts):

        #These log probabilities are in coherence with π_Θ (aₜ | sₜ) in the clipped surrogate objective.
        # The old log probabilitites, or  π_Θk(aₜ | sₜ) (prob at k iteration), we get from batch_log_probs. 
        action_probs = self.actor(batch_obs)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted log probs log_probs
        return log_probs

    #Note: what to do if action_probs are all equal 0? we will need for the opponent to do his turn, will this work out in the env?
    #Calculate the log probabilities of batch actions using most recent actor network.
    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.int)
        
        #Get valid actions, [0] = dice, [1] = action
        valid_actions = [i[1] for i in self.env.get_valid_actions()]

        #If there are no valid actions, dont mask since all actions will become 0. The env will skip the turn if there are no valid actions
        if not valid_actions:
            valid_action_prob = self.actor(obs)
        else:
            action_probs = self.actor(obs)
            valid_action_prob = self.action_mask(valid_actions, action_probs)
            valid_action_prob = torch.tensor(valid_action_prob, dtype= torch.float)
        
        dist = Categorical(valid_action_prob)

        # Sample an action from the distribution and get its log prob
        action_net = dist.sample()
        valid_log_prob = dist.log_prob(action_net)
        
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.

        #why does item work but not numpy??
        #print("action item", action.dtype)
        #print("action numpy", action.numpy())

        #Map output from actor network to the correct action for our environment, which is src, dst = action
        action_env = np.unravel_index(action_net, (8, 8))

        #print("get_action:", action_env)

        return action_net, action_env, valid_log_prob.detach()
    
   
    #Recalculate the probabilities to ensure only valid actions can be chosen.
    def action_mask(self, valid_actions, action_prob):

        valid_net_actions = []
        sum_of_exp = 0
        action_prob = action_prob.detach().numpy()
        valid_probs = list(action_prob)

        #Convert valid environment actions into network actions
        for valid_act in valid_actions:
            valid_net_actions.append(np.ravel_multi_index(valid_act, (8,8)))
        
        #Calculate the sum of valid prob exponents
        for valid_net_act in valid_net_actions:
          sum_of_exp += np.exp(action_prob[valid_net_act])

        #The masking function is Y_k = exp(P_k) / sum(exp(P_valids))
        #Calculate the corrrect masking value for each valid action


        #Loop through each valid_net_action, which are the indexes of the valid actions, and mask each valid probaility, so the sum of the valid actions become 1
        for idx in valid_net_actions:
            valid_probs[idx] = np.exp(action_prob[idx]) / sum_of_exp

        #Loop through and set every probaility which is not valid to 0
        non_valid_net_actions = [i for i in range(64) if i not in valid_net_actions]

        for idx in non_valid_net_actions:
            valid_probs[idx] = 0.0
               
        return valid_probs