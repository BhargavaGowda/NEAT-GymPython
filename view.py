import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle

env = gym.make("BipedalWalker-v3",hardcore=False, render_mode="human")
env = FlattenObservation(env)
observation, info = env.reset()
print(env.action_space)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config.txt")


with open("bestGenome.pkl", "rb") as f:
    genome = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(genome, config)




for _ in range(500):


    action = net.activate(observation) # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        individual.configure_new(config.genome_config)
        net = neat.nn.FeedForwardNetwork.create(individual, config)


env.close()