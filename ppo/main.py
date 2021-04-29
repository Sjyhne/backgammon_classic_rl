import gym
import sys
import torch

from ppo import PPO
from network import feedforwardNN

def train(env, hyperparameters, actor_model, critic_model, timesteps):

    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(env=env, **hyperparameters)

    '''# Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)'''


    # Train the PPO model with a specified total timesteps
    model.learn(total_timesteps = timesteps)


if __name__ == '__main__':
    env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

    hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': False,
				'render_every_i': 10
			  }
    #train(env=env, hyperparameters=hyperparameters, actor_model= "ppo_actor.pth", critic_model="ppo_critic.pth", timesteps= 200)

    model = PPO(env)
    model.learn(100)