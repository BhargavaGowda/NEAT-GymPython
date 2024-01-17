import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle

env = gym.make("BipedalWalker-v3",hardcore=False)
env = FlattenObservation(env)
observation, info = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config.txt")


def fitFunc(genomes,config):

    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        observation, info = env.reset()

        for _ in range(500):

            action = net.activate(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward

            if terminated or truncated:
                break

        genome.fitness = fitness



        


def main():

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    #pop.add_reporter(neat.Checkpointer(5))

    best = pop.run(fitFunc,100)

    with open("bestGenome.pkl", "wb") as f:
        pickle.dump(best, f)
        f.close()

    env.close()

main()