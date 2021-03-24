import gym
import random

env = gym.make('reduced_backgammon_gym:reducedBackgammonGym-v0')

done = False

obs, current_agent = env.reset()

BLACK = 1
WHITE = 0

COLORS = {WHITE: "White", BLACK: "Black"}

class RandomAgent:
    def __init__(self):
        ...

    def apply_random_action(self, environment):
        num_actions = environment.get_n_actions()
        executed = False
        obs = environment.gym.get_current_observation(env.current_agent)

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
                    print("EXECUTED:", action)
                    break
                else:
                    acts.remove(action)
                    c += 1

            if c == len(acts):
                break
        
        return obs, done, winner


env.render()

random_agent = RandomAgent()

while not done:

    print("Current agent:", COLORS[env.current_agent])

    if env.current_agent == BLACK:
        obs, done, winner = random_agent.apply_random_action(env)
        env.render()

    else:
        for _ in range(env.get_n_actions()):

            available_actions = env.get_valid_actions()

            if len(available_actions) == 0:
                break
            
            print(env.gym.non_used_dice)
            for idx, i in enumerate(available_actions):
                print(idx + 1, ":", i)

            action_index = int(input("Input index of preferred action"))

            action = available_actions[action_index - 1][1]

            obs, rew, done, winner, executed = env.step(action)

            env.render()

    env.change_player_turn()

    



