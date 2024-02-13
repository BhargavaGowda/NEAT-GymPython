import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle
import numpy as np
import visualize

# env = gym.make(
#     "LunarLander-v2",
#     continuous = True,
#     gravity = -10.0,
#     enable_wind = False,
#     wind_power = 10.0,
#     turbulence_power = 1.5,
#     render_mode = "human"
# )
env = gym.make("BipedalWalker-v3", render_mode = "human")
env = FlattenObservation(env)
observation, info = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config.txt")


with open("lastPopCrit.pkl", "rb") as f:
    pop = pickle.load(f)

fitness = 0

runNum = 0

for genome in pop:

    visualize.draw_net(config, genome, True)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    observation, info = env.reset()

    for _ in range(200):
        output = net.activate(observation)
        action = np.zeros(env.action_space.shape)
        for i in range(len(action)):
            action[i] = output[2*i]-output[2*i+1]
        observation, reward, terminated, truncated, info = env.step(action)
        fitness+=reward
        if terminated or truncated:
            print("Individual:",runNum,"Fitness",fitness)
            runNum+=1
            fitness = 0
            break


env.close()