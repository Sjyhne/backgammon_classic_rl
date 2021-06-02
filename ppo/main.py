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
    total_wins = model.learn(batches * hyperparameters["episodes_per_batch"])
    return total_wins


def test(env, hyperparameters, actor_model, critic_model, episodes):

    # Create a model for PPO.
    model = PPO(env=env, **hyperparameters)
    
    print(f"Testing", flush=True)
    print(f"Loading in {actor_model} and {critic_model}...", flush=True)
    model.actor.load_state_dict(torch.load(actor_model))
    model.critic.load_state_dict(torch.load(critic_model))
    print(f"Successfully loaded.", flush=True)

    total_wins = []

    for i in range(episodes):
        _, _, _, _, _, _, winner = model.game_loop_vs_random()
        if winner != None:
            total_wins.append(winner)
    
    print("Win percentage:", round(sum(total_wins)/len(total_wins), 3))



if __name__ == '__main__':
    env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

    hyperparameters = {
				'episodes_per_batch': 10, 
				'max_t_per_episode': 500, 
				'gamma': 0.99, 
				'updates_per_iteration': 8,
				'lr': 0.00002, 
				'clip': 0.2,
				'render': False,
			  }




    total_wins = train(env=env, hyperparameters=hyperparameters, actor_model= "ppo_actor2.pth", critic_model="ppo_critic2.pth", batches=35000)

    test(env=env, hyperparameters=hyperparameters, actor_model= "ppo_actor2.pth", critic_model="ppo_critic2.pth", episodes=10000)
    
    with open("BatchAverageEntropy2.txt", "r+") as f:        
        x = f.readlines()

        y = [float(item) for item in x]

    with open("BatchAverageValidEntropy2.txt", "r+") as f:        
        a = f.readlines()
 
        s = [float(item) for item in a]

    plt.figure(0)
    plt.plot(y, label="Average Entropy")
    plt.plot(s, label="Valid Average Entropy")
    plt.title("Average batch entropy")
    plt.legend()
    plt.savefig("./BatchAverageEntropy2.png")

    with open("CriticLosses2.txt", "r+") as f:
        t = f.readlines()

        loss = [float(item) for item in t]

    
    plt.figure(1)
    plt.plot(loss)
    plt.title("Critic loss")
    plt.savefig("./CriticLoss2.png")
    
    with open("ActorLosses2.txt", "r+") as f:        
        r = f.readlines()

        Aloss = [float(item) for item in r]

    plt.figure(2)
    plt.plot(Aloss)
    plt.title("Actor loss")
    plt.savefig("./ActorLoss2.png")

    with open("BatchWinPercentages2.txt", "r+") as f:        
        w = f.readlines()

        wins = [float(item) for item in w]
    
    average_batch_wins = []
    
    chunks = [wins[x:x+100] for x in range(0, len(wins), 1000)]

    for chunk in chunks:
        average_batch_wins.append(sum(chunk)/len(chunk))

    print(average_batch_wins)
            
    plt.figure(3)
    plt.plot(wins)
    plt.title("Average batch wins")
    plt.savefig("./BatchWinPercentages2.png")

    plt.figure(4)
    plt.plot(average_batch_wins)
    plt.title("Average batch wins")
    plt.savefig("./AvgBatchWinPercentages2.png")
    