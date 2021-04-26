import torch
import torch.nn as nn
import numpy as np
import gym
from torch import tensor
from torch.distributions import Categorical
from torch.optim import Adam
from network import feedforwardNN


class PPO:
    def __init__(self, env):
        
        #Initialize hyperparamters
        self._init_hyperparamters()

        #Get environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        #Initalize the actor and the critic
        self.actor = feedforwardNN(self.obs_dim, self.act_dim)
        self.critic = feedforwardNN(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)


    #Learning loop
    def learn(self, t_total):
        t_iterated = 0 #Timetsteps iterated so far

        while t_iterated < t_total:

            batch_obs, batch_actions, batch_log_probs, batch_qvals, batch_lens = self.rollout()

            print("batch obs size", batch_obs.shape)

            print("batch qval size", batch_qvals.shape)

            # Calculate how many timesteps we collected this batch   
            t_iterated += np.sum(batch_lens)

            #Calculate V
            V = self.evaluate_values(batch_obs)
            print("batch value size", V)

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
        
    #Define hyperparameters
    def _init_hyperparamters(self):
        #Default values, need to experiment with changes
        self.t_per_batch = 4000                 #Timesteps per batch
        self.max_t_per_episode = 1000            #Timesteps per episode
        self.gamma = 0.95                       #Gamma for discounted return
        self.updates_per_iteration = 5          #Amount of updates per epoch
        self.clip = 0.2                         #Clip recommended by paper
        self.lr = 0.005                         #Learning rate of optimizers

    #Rollout function to collect batch data(need to understand why we use log on porobabilties)
    def rollout(self):
        batch_obs = []          #Batch observations
        batch_actions = []      #Batch actions  
        batch_log_probs = []    #log probabilities of each action 
        batch_rews = []         #Batch rewards
        batch_qvals = []        #Batch q values, index 0 will correspond to the q value at timestep 1 in the first epsiode
        batch_lens = []         #Lengths of episodes in batch

        t_iterated = 0 #Timesteps iterated so far in batch

        while t_iterated < self.t_per_batch:

            #Rewards this episode
            episode_rews = []
            
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_t_per_episode):
                
                t_iterated += 1

                #Collect observation
                batch_obs.append(obs)


                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                episode_rews.append(rew)
                batch_log_probs.append(log_prob)
                batch_actions.append(action)

                if done:
                    break

            #Collect episodic length and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(episode_rews)
    
        #Reshape data as tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.int)
        batch_log_probs = torch.tensor(batch_log_probs, dtype= torch.float)

        #Collect return
        batch_qvals = self.compute_qvals(batch_rews)

        #Return the batch data
        return batch_obs, batch_actions, batch_log_probs, batch_qvals, batch_lens

    #Q value function, the index is the timestep. at index 1 it has a high value with the discounted future rewards.
    def compute_qvals(self, batch_rews):
        size = 0
        for eps in batch_rews:
            size += len(eps)
            print("eps size", size)

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
    
    #Calculate the log probabilities of batch actions using most recent actor network.
    def evaluate_log_probs(self, batch_obs, batch_acts):
        
        #These log probabilities are in coherence with π_Θ (aₜ | sₜ) in the clipped surrogate objective.
        # The old log probabilitites, or  π_Θk(aₜ | sₜ) (prob at k iteration), we get from batch_log_probs. 
        action_probs = self.actor(batch_obs)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return log_probs

    def get_action(self, obs):
        
        action_probs = self.actor(obs)
        #print("action probs",action_probs)
        dist = Categorical(action_probs)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        #print("action", action)
        log_prob = dist.log_prob(action)
        #print("log probs", log_prob)
        
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.

        #why does item work but not numpy??
        #print("action item", action.dtype)
        #print("action numpy", action.numpy())

        return action.detach().item(), log_prob.detach()
    

print(torch.min(tensor([-2, 1.2, 3.1, -0.8]), tensor([-1.5, 1.2, 1.5, -0.8])).mean())

env = gym.make("CartPole-v1")
print(env.action_space.n)
print(env.observation_space)

import gym
model = PPO(env)
model.learn(1000)

