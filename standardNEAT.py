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
#     enable_wind = False,
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

    numRolloutsPerEval = 1

    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        worstFitness = 10000
        fitness = 0
        runNum = 1
        observation, info = env.reset()

        while True:

            output = net.activate(observation)
            action = np.zeros(env.action_space.shape)
            for i in range(len(action)):
                action[i] = output[2*i]-output[2*i+1]
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward

            if terminated or truncated:
                runNum+=1
                observation, info = env.reset()
                if fitness< worstFitness:
                    worstFitness = fitness
                fitness = 0
            
            if runNum>numRolloutsPerEval:
                break
                

        genome.fitness = worstFitness



        


def main():

    pop = neat.Population(config)
    # pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-799')
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(50))

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