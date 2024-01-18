import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle
import visualize
import numpy as np

# env = gym.make(
#     "LunarLander-v2",
#     continuous = True,
#     gravity = -10.0,
#     enable_wind = True,
#     wind_power = 10.0,
#     turbulence_power = 1.5
# )
env = gym.make("BipedalWalker-v3", hardcore=False)
env = FlattenObservation(env)
observation, info = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config.txt")


def fitFunc(genomes,config):

    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        actionTotal=0
        observation, info = env.reset()

        for _ in range(500):

            output = net.activate(observation)
            action = [0,0,0,0]
            action[0] = output[0]-output[1]
            action[1] = output[2]-output[3]
            action[2] = output[4]-output[5]
            action[3] = output[6]-output[7]
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            actionTotal += np.linalg.norm(np.array(action))

            if terminated or truncated:
                break

        if actionTotal < 20:
            fitness-= 100
        genome.fitness = fitness



        


def main():

    pop = neat.Population(config)
    # pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-799')
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(100))

    best = pop.run(fitFunc,100)

    with open("bestGenome.pkl", "wb") as f:
        pickle.dump(best, f)
        f.close()

    env.close()


    visualize.draw_net(config, best, True)
    visualize.draw_net(config, best, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    stats.save()
    
main()