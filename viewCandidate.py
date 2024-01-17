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
    enable_wind = True,
    wind_power = 10.0,
    turbulence_power = 1.5,
    render_mode = "human"
)
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

numRuns = 10

fitnessList=np.zeros(numRuns)
fitness = 0
actionTotal=0
runNum = 0
while True:

    output = net.activate(observation)
    action = [0,0]
    action[0] = output[0]
    action[1] = -2*output[1]+2*output[2]
    observation, reward, terminated, truncated, info = env.step(action)
    fitness+=reward
    actionTotal += np.linalg.norm(np.array(action))
    if terminated or truncated:
        observation, info = env.reset()
        print("Run:",runNum,"Fitness and action:",fitness,actionTotal)
        fitnessList[runNum]=fitness
        runNum+=1
        fitness = 0
        actionTotal=0
    
    if runNum>=numRuns:
        break

print("avgFitness:",np.mean(fitnessList))

env.close()