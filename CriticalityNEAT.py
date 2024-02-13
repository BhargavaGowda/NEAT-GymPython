import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle
import visualize
import matplotlib.pyplot as plt
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

popSize = 50
gens = 100

def main():

    #setup
    genomeConfig = config.genome_config
    pop = []
    metric1 = []
    fitnessList = []
    bestFitness = -10000
    bestGenome = None
    fitCurve = np.zeros(gens)
    numRolloutsPerEval = 1

    for i in range(popSize):
        newGenome = neat.DefaultGenome(i)
        newGenome.configure_new(genomeConfig)
        newGenome.fitness = 0
        pop.append(newGenome)

    #Warmup
        
    for i in range(popSize):
        genome = pop[i]

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        fitness = 0
        worstFitness = 10000
        runNum = 1
        step = 0
        observation, info = env.reset()
        obsSize = observation.size
        
        # m1 = np.array([])
        m1 = np.zeros(obsSize)
        m2 = np.zeros(obsSize)
        m3 = np.zeros(obsSize)


        while True:

            output = net.activate(observation)
            action = np.zeros(env.action_space.shape)
            for i in range(len(action)):
                action[i] = 2*output[2*i]-2*output[2*i+1]
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                    reward = 0
            step+=1
            fitness += reward
            m1 += 0.1*(observation-m1)
            m2 += 0.01*(observation-m2)
            m3 += 0.001*(observation-m3)

            # if step%20 == 0:
            #     m1 = np.concatenate((m1,observation,[fitness]))


            if step>300:
                runNum+=1
                
                if fitness< worstFitness:
                    worstFitness = fitness
                fitness = 0
                step = 0
                
                observation, info = env.reset()
            
            if runNum>numRolloutsPerEval:
                break


        # print(m1)
        # print(m2)
        # print(m3)
        metric1.append(np.concatenate((m1,m2,m3)))
        fitnessList.append(fitness)


    # for i in range(min(10,popSize)):
    #     print(pop[i].fitness,metric1[i])


    for gen in range(gens):

        print("running gen:", gen)
        for g in range(popSize):

            parent1 = pop[g]
            parent2 = None
            parent2index = 0
            bestCriticality = 0
            for g2 in range(popSize):
                if g!=g2:
                    genomicDist = parent1.distance(pop[g2],genomeConfig)
                    behaviourDist = np.linalg.norm(np.array(metric1[g])-np.array(metric1[g2]))
                    criticality = behaviourDist/genomicDist
                    if criticality>bestCriticality:
                        parent2 = pop[g2]
                        parent2index = g2
                        bestCriticality = criticality
            
            #print("p1:",g,"p2:",parent2index)
            if not parent2:
                raise RuntimeError("no parent2")
            



            testGenome = neat.DefaultGenome(g)
            testGenome.configure_crossover(parent1,parent2,genomeConfig)
            testGenome.mutate(genomeConfig)
            testGenome.fitness = 0

            pop[g] = testGenome

            net = neat.nn.FeedForwardNetwork.create(testGenome, config)
            fitness = 0
            worstFitness = 10000
            step = 0
            runNum = 1
            observation, info = env.reset()
            # m1 = np.array([])
            m1 = np.zeros(obsSize)
            m2 = np.zeros(obsSize)
            m3 = np.zeros(obsSize)

            while True:

                output = net.activate(observation)
                action = np.zeros(env.action_space.shape)
                for i in range(len(action)):
                    action[i] = 2*output[2*i]-2*output[2*i+1]
                observation, reward, terminated, truncated, info = env.step(action)
                step+=1
                if terminated or truncated:
                    reward = 0
                fitness += reward
                m1 += 0.1*(observation-m1)
                m2 += 0.01*(observation-m2)
                m3 += 0.001*(observation-m3)
                # if step%20 == 0:
                #     m1 = np.concatenate((m1,observation,[fitness]))
                
                if step>300:
                    runNum+=1
                    
                    if fitness< worstFitness:
                        worstFitness = fitness
                    fitness = 0
                    step = 0
                    observation, info = env.reset()

                if runNum>numRolloutsPerEval:
                    break     

            metric1[g] = np.concatenate((m1,m2,m3))
            fitnessList[g] = worstFitness
        

            if worstFitness>bestFitness:
                bestFitness = worstFitness
                bestGenome = testGenome
                bestGenomeM1 = m1

        #reporter
        print("avg fitness", np.mean(fitnessList) ,"best Fitness:", bestFitness)
        # for i in range(min(10,popSize)):
        #     print(i)
        fitCurve[gen] = bestFitness
        
        if gen%100 == 0 and gen>0:
            with open("gen"+str(gen)+"_CheckpointCrit.pkl", "wb") as f:
                pickle.dump(pop, f)




    print("BestFitness:", bestFitness)
    visualize.draw_net(config, bestGenome, True)

    plt.plot(fitCurve)
    plt.show()
    

    with open("bestGenomeCrit.pkl", "wb") as f:
        pickle.dump(bestGenome, f)


    with open("lastPopCrit.pkl", "wb") as f:
        pickle.dump(pop, f)


    

main()