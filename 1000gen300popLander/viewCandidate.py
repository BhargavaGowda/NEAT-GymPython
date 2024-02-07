import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle
import numpy as np
import visualize

env = gym.make(
    "LunarLander-v2",
    continuous = True,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 10.0,
    turbulence_power = 1.5,
    render_mode = "human"
)
# env = gym.make("BipedalWalker-v3", render_mode = "human")
env = FlattenObservation(env)
observation, info = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config.txt")


with open("bestGenome.pkl", "rb") as f:
    genome = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(genome, config)
print(genome.size())
visualize.draw_net(config, genome, True)

numRuns = 20

fitnessList=np.zeros(numRuns)
fitness = 0
runNum = 0
while True:

    output = net.activate(observation)
    action = np.zeros(env.action_space.shape)
    action = [output[0],0]
    action[1] = 2*output[2]-2*output[1]
    observation, reward, terminated, truncated, info = env.step(action)
    fitness+=reward
    if terminated or truncated:
        observation, info = env.reset()
        print("Run:",runNum,"Fitness:",fitness)
        fitnessList[runNum]=fitness
        runNum+=1
        fitness = 0
    
    if runNum>=numRuns:
        break

print("maxFitness:",np.max(fitnessList))

env.close()