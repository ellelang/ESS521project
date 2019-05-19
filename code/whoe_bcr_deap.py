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

plt.style.use('bmh')
wcmodatawhole = pd.read_csv(data_folder/'output/wcmogawhole.csv')
wcmodatawhole.shape
near_wld_whole = wcmodatawhole[['Site_ID','SedRed','Cost','NEAR_FID', 'NEAR_DIST','area_m2']].sort_values(by=['NEAR_FID','SedRed'])
near_wld_whole
near_wld_whole["Sedrank"] = near_wld_whole.groupby("NEAR_FID")["SedRed"].rank("dense", ascending=False)
near_wld_whole["Sedrank"]
near_wld_whole["bcr"] = near_wld_whole.SedRed/near_wld_whole.Cost
near_wld_whole["bcr"]

def countX(lst, x): 
    return lst.count(x)
clid = list(near_wld_whole['NEAR_FID'])
clid
lst = clid

near_wld_whole['Cluster_size'] = [countX(lst, x) for x in clid]
near_wld_whole['Cluster_size']

NBR_ITEMS = 7874
top = np.arange(10,7874,10)
top
range(len(top))
topname = ["Top" + str(i) for i in top]
topname

for i in range (len(top)):
    near_wld_whole[topname[i]] = [0]* NBR_ITEMS

for t in range(len(top)):
    near_wld_whole.loc[near_wld_whole.nlargest(top[t],'bcr').index,topname[t]] = 1 

near_wld_whole.to_csv(data_folder/'output/bcr_ranking_whole.csv', index = False)

sed = near_wld_whole['SedRed']
cost = near_wld_whole['Cost']
len(cost)
len(sed)
seeds_array = np.array(near_wld_whole[topname])
seeds_array
seeds_array_to_list = seeds_array.tolist()
json_file = "bcrseeds_whole.json" 
json.dump(seeds_array_to_list, open(data_folder/json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)


#########add interactions
topname_epis = ["Top_epis" + str(i) for i in top]
topname_epis
cluster_size = near_wld_whole['Cluster_size']
sedrank =  near_wld_whole['Sedrank']
for i in range (len(top)):
    near_wld_whole[topname_epis[i]] = [0]* NBR_ITEMS
    
bcr_epis = [0]*NBR_ITEMS
len(bcr_epis)
sed_epis = [None]*NBR_ITEMS

for i in range(NBR_ITEMS):
    topn = int(np.ceil(cluster_size[i]/2))
    if sedrank[i] > topn:
        bcr_epis[i] = 0
        sed_epis[i] = 0
    else:
        bcr_epis[i] = sed[i]/cost[i]
        sed_epis[i] = sed[i]

bcr_epis
near_wld_whole['bcr_epis'] = bcr_epis
near_wld_whole['bcr_epis']
near_wld_whole['SedRed_epis'] = sed_epis
#topname_epis
for t in range(len(top)):
    near_wld_whole.loc[near_wld_whole.nlargest(top[t],'bcr_epis').index,topname_epis[t]] = 1

sedsum_epis = [sum(sed_epis * near_wld_whole[i]) for i in topname_epis]
sedsum_epis 
costsum_epis = [sum(cost * near_wld_whole[i]) for i in topname_epis]
costsum_epis

sedsum = [sum(sed_epis * near_wld_whole[i]) for i in topname]

costsum = [sum(cost * near_wld_whole[i]) for i in topname]

plt.scatter(sedsum, costsum, c='b', marker='x', label='bcr_ranking')
plt.scatter(sedsum_epis,costsum_epis, c = 'c', marker = 'o', label='bcr_epistasis')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.show()

##########EA


parcelid =  near_wld_whole.Site_ID
sed = near_wld_whole.SedRed
cost = near_wld_whole.Cost
clusts = near_wld_whole.NEAR_FID
clusterids = list(near_wld_whole.NEAR_FID.unique())
clusterids
sed
clustersize = near_wld_whole.Cluster_size
sedrank = near_wld_whole.Sedrank


creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)


items = {}
for i in range(NBR_ITEMS):
    items[i] = (cost[i], sed[i],clusts[i],clustersize[i],sedrank[i])

items.values()



#creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
#creator.create("Individual", set, fitness=creator.Fitness)

def initIndividual(icls, content):
    return icls(content)


def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)



IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50


toolbox = base.Toolbox()
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_item, IND_INIT_SIZE)

toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, data_folder/"bcrseeds_whole.json")

toolbox.register("population", tools.initRepeat, list, toolbox.individual)




def evalKnapsack(individual):
    cost_val = 0.0
    sed_val = 0.0
    for i in individual:
        cluster_size = items[i][3]
        topn = int(np.ceil(cluster_size/2))
        sedrank = items[i][4]
        if sedrank > topn:
            cost_val += items[i][0]
            sed_val += 0
        else: 
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


toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def main():
    NGEN = 5000
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    #no seeding
    pop = toolbox.population(n=MU)
    #seeding!!
    #pop = toolbox.population_guess()
    #hof = tools.ParetoFront()
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
#    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, 
                              stats)
    front = np.array([ind.fitness.values for ind in pop])
    
    return pop, front

if __name__ == "__main__":
    results = main()

#############no seeding
noseedfront = results[1]
noseedpop_results = results[0]
noseedpop_results.sort(key=lambda x: x.fitness.values)
noseedpop_results[0]
noseedpop_results[-1]
noseedpop_results

noseedcost_f = noseedfront[:,0]
noseedsed_f = noseedfront[:,1]
#max(noseedsed_f)

##############bcr seeding

def main():
    NGEN = 5000
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    #no seeding
    #pop = toolbox.population(n=MU)
    #seeding!!
    pop = toolbox.population_guess()
    #hof = tools.ParetoFront()
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
#    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, 
                              stats)
    front = np.array([ind.fitness.values for ind in pop])
    
    return pop, front

if __name__ == "__main__":
    bcrseed_results = main()

bcrseed_front = bcrseed_results[1]
bcrseed_pop_results = bcrseed_results[0]
bcrseed_pop_results.sort(key=lambda x: x.fitness.values)
bcrseed_pop_results[0]
bcrseed_pop_results[-1]
bcrseed_pop_results

bcrseed_cost_f = bcrseed_front[:,0]
bcrseed_sed_f = bcrseed_front[:,1]

plt.scatter(sedsum, costsum, c='b', marker='x', label='bcr_ranking')
plt.scatter(sedsum_epis,costsum_epis, c = 'c', marker = 'o', label='bcr_epistasis')
plt.scatter(noseedsed_f, noseedcost_f,c='y', marker='v', label='EA_noseed')
plt.scatter(bcrseed_sed_f, bcrseed_cost_f, c='r', marker='s', label='EA_bcrseed')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.savefig('whole_paretofront_compare.png',dpi = 100)
plt.show()