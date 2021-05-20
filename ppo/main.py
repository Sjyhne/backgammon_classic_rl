import gym
import sys
import torch

import os

from ppo import PPO
from network import feedforwardNN
import matplotlib.pyplot as plt

def train(env, hyperparameters, actor_model, critic_model, batches):

    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if os.path.exists(actor_model) and os.path.exists(critic_model):
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif os.path.exists(actor_model) or os.path.exists(critic_model): # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)




    # Train the PPO model with a specified total of games
    total_wins = model.learn(batches * hyperparameters["episode_per_batch"])
    return total_wins


def test(env, hyperparameters, actor_model, critic_model, episodes):

    # Create a model for PPO.
    model = PPO(env=env, **hyperparameters)

    print(f"Loading in {actor_model} and {critic_model}...", flush=True)
    model.actor.load_state_dict(torch.load(actor_model))
    model.critic.load_state_dict(torch.load(critic_model))
    print(f"Successfully loaded.", flush=True)

    total_wins = []

    for i in range(episodes):
        _, _, _, _, winner = model.game_loop_vs_random()
        if winner != None:
            total_wins.append(winner)
    
    print("Win percentage:", round(sum(total_wins)/len(total_wins), 3))



if __name__ == '__main__':
    env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

    hyperparameters = {
				'episode_per_batch': 100, 
				'max_t_per_episode': 200, 
				'gamma': 0.99, 
				'updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': False,
			  }

    total_wins = train(env=env, hyperparameters=hyperparameters, actor_model= "./ppo_actor.pth", critic_model="./ppo_critic.pth", batches=5)

    test(env=env, hyperparameters=hyperparameters, actor_model= "./ppo_actor.pth", critic_model="./ppo_critic.pth", episodes=300)
    
    print(total_wins)

    plt.plot(total_wins)
    plt.title("Batch win percentages")
    plt.savefig("./total_win_graph.png")
