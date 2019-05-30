from __future__ import (absolute_import, division, print_function)
from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')
import random
import pandas as pd
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import json
import matplotlib.pyplot as plt

maple29data = pd.read_csv(data_folder/'output/prep2_MAP_sub29.csv')
maple29data.columns
maple29data.index
NBR_ITEMS = 468
parcelid =  maple29data.Site_ID
sed = maple29data.SedRed
cost = maple29data.Cost
clusts = maple29data.NEAR_FID
clusterids = list(maple29data.NEAR_FID.unique())
clusterids
sed
[value == 113 for value in clusts]

maple29data.groupby("NEAR_FID")["SedRed"]

clustersize = maple29data.Cluster_size
sedrank = maple29data.Sedrank



creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)
items = {}
for i in range(NBR_ITEMS):
    items[i] = (cost[i], sed[i],clusts[i],clustersize[i],sedrank[i])

items.values()


items

cluster_items = pd.DataFrame( [item for item in items.values() if item[2] == 13690])
cluster_items

def initIndividual(icls, content):
    return icls(content)


def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)



IND_INIT_SIZE = 5000
MAX_ITEM = 50
MAX_WEIGHT = 50


toolbox = base.Toolbox()
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_item, IND_INIT_SIZE)

toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, data_folder/"bcrseeds.json")

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#def evalKnapsack(individual):
#    cost_val = 0.0
#    sed_val = 0.0
#    for i in individual:
#        cluster_size = items[i][3]
#        topn = int(np.ceil(cluster_size/2))
#        sedrank = items[i][4]
#        if sedrank > topn:
#            cost_val += items[i][0]
#            sed_val += 0
#        else: 
#            cost_val += items[i][0]
#            sed_val += items[i][1]
#    
#    return cost_val, sed_val


def evalKnapsack_simple(individual):
    cost_val = 0.0
    sed_val = 0.0
    for i in individual:
        cost_val += items[i][0]
        sed_val += items[i][1]
    
    return cost_val, sed_val

def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2

def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,





toolbox.register("evaluate", evalKnapsack_simple)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)


def main():
    NGEN = 500
    MU = 100
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    #no seeding
    pop = toolbox.population(n=MU)
    #seeding!!
    #pop = toolbox.population_guess()
    hof = tools.ParetoFront()
    #logbook = tools.Logbook()
    #logbook.header = "gen", "evals", "std", "min", "avg", "max"
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
#    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, 
                              stats,halloffame=hof)
    
    
    return pop, stats, hof

if __name__ == "__main__":
    results = main()


#############no seeding
noseedpop = results[0]
noseedstats = results[1]
noseedhof = results[2] 
noseedhof
noseedfront = np.array([ind.fitness.values for ind in noseedhof])
noseedfront
noseedcost_f = noseedfront[:,0]
noseedsed_f = noseedfront[:,1]
#max(noseedsed_f)


##############bcr seeding
    
def main():
    NGEN = 500
    MU = 100
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    #no seeding
    #pop = toolbox.population(n=MU)
    #seeding!!
    pop = toolbox.population_guess()
    hof = tools.ParetoFront()
    #logbook = tools.Logbook()
    #logbook.header = "gen", "evals", "std", "min", "avg", "max"
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
#    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)
    #front = np.array([ind.fitness.values for ind in pop])
    
    return pop, stats, hof

if __name__ == "__main__":
    bcrseed_results = main()

#############Seedings
bcrseed_pop = bcrseed_results[0]
bcrseed_stats = bcrseed_results[1]
bcrseed_hof = bcrseed_results[2] 

bcrseed_front = np.array([ind.fitness.values for ind in bcrseed_hof])
bcrseed_front
bcrseedcost_f = bcrseed_front[:,0]
bcrseedsed_f = bcrseed_front[:,1]

##########
maple29data = pd.read_csv(data_folder/'output/prep2_MAP_sub29.csv')
maple29data.columns
maple29data.index
NBR_ITEMS = 468
top = np.arange(10,469,10)
top
range(len(top))
topname = ["Top" + str(i) for i in top]
topname


for i in range (len(top)):
    maple29data[topname[i]] = [0]* NBR_ITEMS


maple29data.iloc[maple29data.nlargest(3,'bcr').index]

maple29data.loc[maple29data.nlargest(3,'bcr').index,'Top10'] = 1 
maple29data.loc[maple29data.nlargest(3,'bcr').index]['Top10']

maple29data.columns

for t in range(len(top)):
    maple29data.loc[maple29data.nlargest(top[t],'bcr').index,topname[t]] = 1 

#maple29data.to_csv(data_folder/'output/bcr_ranking_MAPsub29.csv', index = False)

sed = maple29data['SedRed']
cost = maple29data['Cost']

sum(sed * maple29data['Top10'])

sedsum = [sum(sed * maple29data[i]) for i in topname]

costsum = [sum(cost * maple29data[i]) for i in topname]




plt.scatter(sedsum, costsum, c='b', marker='x', label='bcr_ranking')
plt.scatter(noseedsed_f, noseedcost_f,c='y', marker='v', label='EA_noseed')
plt.scatter(bcrseedsed_f, bcrseedcost_f,c='r', marker='s', label='EA_bcrseed')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.show()



