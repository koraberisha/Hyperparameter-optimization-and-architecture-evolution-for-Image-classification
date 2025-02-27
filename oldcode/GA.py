import random
import tensorTest
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

POP_SIZE        = 4     # population size
GENOME_SIZE     = 20    # number of bits per individual in the population
GENERATIONS     = 20  # maximal number of generations to run GA
TOURNAMENT_SIZE = 1    # size of tournament for tournament selection
PROB_MUTATION   = 1/POP_SIZE   # bitwise probability of mutatio


class CurrentStatistics:
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []

    def addx(self,xval):
        self.x.append(xval)

    def addy(self,yval):
        self.y.append(yval)

    def addz(self,zval):
        self.z.append(zval)


global stats
global ticks
ticks = 0
stats = CurrentStatistics()


def plot_threed(stats):

    print("plott")

    fig, ax = plt.subplots(subplot_kw={"projection": "2d"})
    # Make data.
    X = np.array([1,2,3,4,5]) # num of epochs
    Y = np.array(stats.z)  # current generation

    print(X)
    print(Y)
    # Plot the surface.
    surf = ax.plot(X, Y)
    plt.show()

def init_population(): # population comprises bitstrings
    return([(''.join(random.choice("01") for i in range(GENOME_SIZE)))
                        for i in range(POP_SIZE)])

def fitness(individual): # fitness = number of 1s in individual
    out1,conv1,acti1,pool2,lchoice = convert_bitstring(individual)
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print(individual)
    fitnessfor = tensorTest.compile_model(out1, conv1, acti1, 0, 0, 0, 0, 0, 0, 0, pool2, lchoice)

    print(fitnessfor)
    return fitnessfor[len(fitnessfor)-1]

def convert_bitstring(bitstring):
    out1 = bitstring[0:2]
    conv1 = bitstring[3:5]
    acti2 = bitstring[6:8]
    pool2 = bitstring[9:11]
    layerchoice = bitstring[12:19]



    return int(str(out1),2),int(str(conv1),2),int(str(acti2),2),int(str(pool2),2),layerchoice


def selection(population): # select one individual using tournament selection
    tournament = [random.choice(population) for i in range(TOURNAMENT_SIZE)]
    fitnesses = [fitness(tournament[i]) for i in range(TOURNAMENT_SIZE)]
    return tournament[fitnesses.index(max(fitnesses))]

def crossover(parent1,parent2): # single-point crossover
    parent1,parent2=str(parent1),str(parent2)
    xo_point=random.randint(1, GENOME_SIZE-1)
    return([
            parent1[0:xo_point]+parent2[xo_point:GENOME_SIZE],
            parent2[0:xo_point]+parent1[xo_point:GENOME_SIZE] ])

def bitflip(bit): # used in mutation
    bit=str(bit)
    if bit == "0":
        return "1"
    else:
        return "0"

def mutation(individual): # bitwise mutation with probability PROB_MUTATION
    individual=str(individual)
    for i in range(GENOME_SIZE):
        if random.random() < PROB_MUTATION:
            individual = individual[:i] + bitflip(i) + individual[i+1:]
    return(individual)

def print_population(population):
    fitnesses=[fitness(population[i]) for i in range(POP_SIZE)]
    print(list(zip(population,fitnesses)))


# begin GA run
random.seed() # initialize internal state of random number generator
population=init_population() # generation

oldmax = 0

vals = []
for gen in range(GENERATIONS):
    print("Generation ",gen)
    print_population(population)
    for i in range(0,POP_SIZE):
        vals.append(fitness(population[i]))

        print(vals)

        if max(vals) > oldmax:
            break;
            oldmax = max(vals)
        stats.z.append(vals)


    nextgen_population=[]




    for i in range(int(POP_SIZE/2)):
        parent1=selection(population)
        parent2=selection(population)
        offspring=crossover(parent1,parent2)
        nextgen_population.append(mutation(offspring[0]))
        nextgen_population.append(mutation(offspring[1]))



    population=nextgen_population
