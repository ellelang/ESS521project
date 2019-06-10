from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import array
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

#plt.style.use('bmh')
wcmodata = pd.read_csv(data_folder/'output/wcmogawhole.csv')
wcmodata.shape
wcmodata0 = wcmodata[['Site_ID','SedRed','Cost','NEAR_FID','NEAR_DIST','area_m2']].sort_values(by=['NEAR_FID','SedRed'])
wcmodata0
wcmodata0["Sedrank"] = wcmodata0.groupby("NEAR_FID")["SedRed"].rank("dense", ascending=False)
wcmodata0["Sedrank"]
wcmodata0["NEAR_DIST"]
wcmodata0["NEARrank"] = wcmodata0.groupby("NEAR_FID")["NEAR_DIST"].rank("dense", ascending=True)
wcmodata0["NEARrank"]

wcmodata0["bcr"] = wcmodata0.SedRed/wcmodata0.Cost
wcmodata0["bcr"]
sed = wcmodata0['SedRed']
cost = wcmodata0['Cost']

wcmodata0.to_csv(data_folder/'output/save_groupby_bcr_ranking_WCMO.csv', index = False)

near_wld = pd.read_csv(data_folder/'output/save_groupby_bcr_ranking_WCMO.csv')

near_wld.columns


def countX(lst, x): 
    return lst.count(x)
clid = list(near_wld['NEAR_FID'])
clid
lst = clid

near_wld['Cluster_size'] = [countX(lst, x) for x in clid]
near_wld['Cluster_size']

NBR_ITEMS = 7874
top = np.arange(10,7874,10)
topname = ["Top" + str(i) for i in top]
topname

for t in range(len(top)):
    near_wld[topname[t]] = [0]* NBR_ITEMS
    near_wld.loc[near_wld.nlargest(top[t],'bcr').index,topname[t]] = 1

sed_noepis = [sum(sed * near_wld[i]) for i in topname]
cost_noepis  = [sum(cost * near_wld[i]) for i in topname]

plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_sed')

####################

##save seeds

##########save seeds
seedslist = []
for i in range(len(top)):
    indexvalues = near_wld.index[near_wld[topname[i]] == 1].tolist()
    seedslist.append(indexvalues)
np.array(seedslist)
json_file = "bcrseeds_wcmo.json" 
json.dump(seedslist, open(data_folder/json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)


#########add interactions

topname_epis = ["Top_epis" + str(i) for i in top]
topname_epis
cluster_size = near_wld['Cluster_size']
sedrank =  near_wld['Sedrank']
disrank = near_wld['NEARrank']
bcr_epis = [None]*NBR_ITEMS
sed_epis = [None]*NBR_ITEMS

for i in range(NBR_ITEMS):
    topn = int(np.ceil(cluster_size[i]/2))
    if disrank[i] > topn:
        bcr_epis[i] = 0
        sed_epis[i] = 0
    else:
        sed_epis[i] = near_wld['SedRed'][i]
        bcr_epis[i] = sed_epis[i]/near_wld['Cost'][i]

bcr_epis
near_wld['bcr_epis'] = bcr_epis
near_wld['bcr_epis']
near_wld['SedRed_epis'] = sed_epis



for t in range(len(top)):
    near_wld[topname_epis[t]] = [0]* NBR_ITEMS
    near_wld.loc[near_wld.nlargest(top[t],'bcr_epis').index,topname_epis[t]] = 1
#near_wld.loc[near_wld.nlargest(top[t],'bcr_epis').index,topname_epis[t]] = 1    

sedsum_epis = [sum(sed_epis * near_wld[i]) for i in topname_epis]
sedsum_epis 
costsum_epis = [sum(cost * near_wld[i]) for i in topname_epis]
costsum_epis
plt.scatter(sedsum_epis,costsum_epis, c = 'c', marker = 'o', label='bcr_ranking_epistasis')

near_wld.to_csv(data_folder/'output/bcr_ranking_wcmo608.csv', index = False)


sedsum_ignore_epis = [sum(sed_epis * near_wld[i]) for i in topname]
sedsum_ignore_epis 
costsum_ignore_epis = [sum(cost * near_wld[i]) for i in topname]
costsum_ignore_epis
#plt.scatter(sedsum_ignore_epis,costsum_ignore_epis, c = 'b', marker = 'o', label='bcr_ranking_epistasis')

#plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')
plt.scatter(sedsum_ignore_epis, costsum_ignore_epis, c='b', marker='x', label='ignore_epistasis')
plt.scatter(sedsum_epis,costsum_epis, c = 'c', marker = 'o', label='consider_epistasis')
plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.savefig('wcmo_bcr_epis_withnoepis.png', dpi=300)
plt.show()

################EA


###############
##########EA
parcelid =  near_wld.Site_ID
sed = near_wld.SedRed
cost = near_wld.Cost
clusters = near_wld.NEAR_FID
clusterids = list(near_wld.NEAR_FID.unique())
clusterids
sed
clustersize = near_wld.Cluster_size
sedrank = near_wld.Sedrank


creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

items = {}
for i in range(NBR_ITEMS):
    items[i] = (cost[i], sed[i],clustersize[i],disrank[i])

items.values()

def initIndividual(icls, content):
    return icls(content)


def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)


IND_INIT_SIZE = 1000
toolbox = base.Toolbox()
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_item, IND_INIT_SIZE)

toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, data_folder/"bcrseeds_wcmo.json")

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack_simple(individual):
    cost_val = 0.0
    sed_val = 0.0
    for i in individual:
        cost_val += items[i][0]
        sed_val += items[i][1]
    return cost_val, sed_val
        
    
#
def evalKnapsack(individual):
    cost_val = 0.0
    sed_val = 0.0
    for i in individual:
        cluster_size = items[i][2]
        topn = int(np.ceil(cluster_size/2))
        disrank = items[i][3]
        if disrank > topn:
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
    NGEN = 1000
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
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)
    #front = np.array([ind.fitness.values for ind in pop])
    
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

###############
##############bcr seeding

def main():
    NGEN = 5000
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


bcrseed_pop = bcrseed_results[0]
bcrseed_stats = bcrseed_results[1]
bcrseed_hof = bcrseed_results[2] 

bcrseed_front = np.array([ind.fitness.values for ind in bcrseed_hof])
bcrseed_front
bcrseedcost_f = bcrseed_front[:,0]
bcrseedsed_f = bcrseed_front[:,1]





plt.scatter(noseedsed_f, noseedcost_f,c='y', marker='v', label='EA_noseed')
plt.scatter(bcrseedsed_f, bcrseedcost_f, c='r', marker='s', label='EA_bcrseed')
plt.scatter(sedsum_ignore_epis, costsum_ignore_epis, c='b', marker='x', label='bcr_ignore_epistasis')
plt.scatter(sedsum_epis,costsum_epis, c = 'c', marker = 'o', label='bcr_consider_epistasis')
plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.savefig('wcmopf_episseedEA.png', dpi=300)
#plt.savefig('wcmopf_ignore.png', dpi=300)
#plt.savefig('wcmopf_consider.png', dpi=300)
#plt.savefig('wcmopf_noepis.png', dpi=300)

plt.show()
