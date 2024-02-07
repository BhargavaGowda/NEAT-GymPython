import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle
import visualize
import matplotlib.pyplot as plt
import numpy as np


env = gym.make(
    "LunarLander-v2",
    continuous = True,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 10.0,
    turbulence_power = 1.5
)
# env = gym.make("BipedalWalker-v3", hardcore=False)
env = FlattenObservation(env)
observation, info = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    "config.txt")

popSize = 50
gens = 300

def main():

    #setup
    genomeConfig = config.genome_config
    pop = []
    metric1 = []
    fitnessList = []
    bestFitness = -10000
    bestGenome = None
    bestGenomeM1 = None
    fitCurve = np.zeros(gens)
    numRolloutsPerEval = 3

    for i in range(popSize):
        newGenome = neat.DefaultGenome(i)
        newGenome.configure_new(genomeConfig)
        newGenome.fitness = 0
        pop.append(newGenome)

    #Warmup
        
    for i in range(popSize):
        genome = pop[i]

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        m1=0
        fitness = 0
        worstFitness = 10000
        runNum = 1
        observation, info = env.reset()

        while True:

            output = net.activate(observation)
            action = np.zeros(env.action_space.shape)
            for i in range(len(action)):
                action[i] = 2*output[2*i]-2*output[2*i+1]
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward

            if terminated or truncated:
                runNum+=1
                observation, info = env.reset()
                if fitness< worstFitness:
                    worstFitness = fitness
                fitness = 0
                x = observation[0]
                if x <-0.3:
                    m1 = 0
                elif x<0.3:
                    m1 = 1
                else:
                    m1 = 2
            
            if runNum>numRolloutsPerEval:
                break

        fitnessList.append(worstFitness)
        metric1.append(m1)


    for i in range(min(10,popSize)):
        print(pop[i].fitness,metric1[i])


    for gen in range(gens):

        print("running gen:", gen)
        for g in range(popSize):

            parent1 = pop[g]
            parent2 = None
            parent2index = 0
            closestDist = 10000
            for g2 in range(popSize):
                if g!=g2:
                    dist = parent1.distance(pop[g2],genomeConfig)
                    if dist<closestDist:
                        parent2 = pop[g2]
                        parent2index = g2
                        closestDist = dist
            
            #print("p1:",g,"p2:",parent2index)
            if not parent2:
                raise RuntimeError("no parent2")


            testGenome = newGenome = neat.DefaultGenome(g)
            testGenome.configure_crossover(parent1,parent2,genomeConfig)
            testGenome.mutate(genomeConfig)

            net = neat.nn.FeedForwardNetwork.create(testGenome, config)
            fitness = 0
            worstFitness = 10000
            runNum = 1
            m1=0
            observation, info = env.reset()

            while True:

                output = net.activate(observation)
                action = np.zeros(env.action_space.shape)
                for i in range(len(action)):
                    action[i] = 2*output[2*i]-2*output[2*i+1]
                observation, reward, terminated, truncated, info = env.step(action)
                fitness += reward
                if terminated or truncated:
                    runNum+=1
                    observation, info = env.reset()
                    if fitness< worstFitness:
                        worstFitness = fitness
                    fitness = 0
                    x = observation[0]
                    if x<-0.3:
                        m1 = 0
                    elif x<0.3:
                        m1 = 1
                    else:
                        m1 = 2
            
                if runNum>numRolloutsPerEval:
                    break
            
            #print("p1 metric:", metric1[g], "p2 metric:", metric1[parent2index], "test action total:", m1)
            # novelty = (np.linalg.norm(m1-metric1[g])+np.linalg.norm(m1-metric1[parent2index]))/2
            #print("novelty:", novelty)
            if m1 != metric1[g] and m1 != metric1[parent2index]:
                testGenome.fitness = 100
            else:
                testGenome.fitness = 0
                

            #Note: this fitness is novelty
            if testGenome.fitness>0:
                pop[g] = testGenome
                metric1[g] = m1
                #not this
                fitnessList[g] = worstFitness

            if worstFitness>bestFitness:
                bestFitness = worstFitness
                bestGenome = testGenome
                bestGenomeM1 = m1

        #reporter
        print("avg fitness", np.mean(fitnessList) ,"best Fitness:", bestFitness)
        for i in range(min(10,popSize)):
            print(i,metric1[i])
        fitCurve[gen] = bestFitness
        #discount novelty every generation
        for i in range(popSize):
            pop[i].fitness*=0.9

        #elitism 1
        pop[0] = bestGenome
        metric1[0] = bestGenomeM1
        fitnessList[0] = bestFitness


    print("BestFitness:", bestFitness)
    visualize.draw_net(config, bestGenome, True)

    plt.plot(fitCurve)
    plt.show()
    

    with open("bestGenomeLNS.pkl", "wb") as f:
        pickle.dump(bestGenome, f)
        f.close()

    with open("lastPopLNS.pkl", "wb") as f:
        pickle.dump(pop, f)
        f.close()

    

main()