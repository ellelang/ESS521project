from deap import tools
from deap import creator
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import benchmarks
from deap import base
from deap import algorithms
import array
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')


# plt.style.use('bmh')
#df = pd.read_csv(data_folder/'output/prep2_MAP_sub29.csv')
# df.shape
# df.columns
#near_wld_maple0 = df[['Site_ID','SedRed','Cost','NEAR_FID', 'NEAR_DIST','area_m2']].sort_values(by=['NEAR_FID','SedRed'])
# near_wld_maple0.columns
#near_wld_maple0["Sedrank"] = near_wld_maple0.groupby("NEAR_FID")["SedRed"].rank("dense", ascending=False)
# near_wld_maple0["Sedrank"]
# near_wld_maple0["NEAR_DIST"]
#near_wld_maple0["NEARrank"] = near_wld_maple0.groupby("NEAR_FID")["NEAR_DIST"].rank("dense", ascending=True)
# near_wld_maple0["NEARrank"]
#
#near_wld_maple0["bcr"] = near_wld_maple0.SedRed/near_wld_maple0.Cost
# near_wld_maple0["bcr"]
#near_wld_maple0.to_csv(data_folder/'output/MAP29save_groupby.csv', index = False)
#

dataset = pd.read_csv(data_folder/'output/MAP29save_groupby.csv')
dataset.shape
NBR_ITEMS = 468


top = np.arange(10, 468, 5)
topname = ["Top" + str(i) for i in top]
topname
for t in range(len(top)):
    dataset[topname[t]] = [0] * NBR_ITEMS
    dataset.loc[dataset.nlargest(top[t], 'bcr').index, topname[t]] = 1

# where is the top 5 largest bcr
dataset.nlargest(5, ['bcr']).index
# where is the top 5 largest SedRed
dataset.nlargest(5, ['SedRed']).index

sed = dataset['SedRed']
#cost = [1] * NBR_ITEMS
cost = dataset['Cost']


sed_noepis = [sum(sed * dataset[i]) for i in topname]
cost_noepis = [sum(cost * dataset[i]) for i in topname]
sed_noepis
#plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_sed')

# save seeds
seedslist = []
for i in range(len(top)):
    indexvalues = dataset.index[dataset[topname[i]] == 1].tolist()
    seedslist.append(indexvalues)
np.array(seedslist)
json_file = "bcrseeds_maple29.json"
json.dump(seedslist, open(data_folder/json_file, 'w',
                          encoding='utf-8'), sort_keys=True, indent=4)

# add interactions


def countX(lst, x):
    return lst.count(x)


clid = list(dataset['NEAR_FID'])
clid
lst = clid

dataset['Cluster_size'] = [countX(lst, x) for x in clid]
dataset['Cluster_size']

topname_epis = ["Top_epis" + str(i) for i in top]
topname_epis
cluster_size = dataset['Cluster_size']
disrank = dataset['NEARrank']

bcr_epis = [None]*NBR_ITEMS
sed_epis = [None]*NBR_ITEMS

for i in range(NBR_ITEMS):
    topn = int(np.ceil(cluster_size[i]/2))
    if disrank[i] > topn:
        bcr_epis[i] = 0
        sed_epis[i] = 0
    else:
        sed_epis[i] = dataset['SedRed'][i]
        bcr_epis[i] = sed_epis[i]/dataset['Cost'][i]

bcr_epis
dataset['bcr_epis'] = bcr_epis
dataset['bcr_epis']
dataset['SedRed_epis'] = sed_epis

for t in range(len(top)):
    dataset[topname_epis[t]] = [0] * NBR_ITEMS
    dataset.loc[dataset.nlargest(
        top[t], 'bcr_epis').index, topname_epis[t]] = 1

sedsum_epis = [sum(sed_epis * dataset[i]) for i in topname_epis]
sedsum_epis
costsum_epis = [sum(cost * dataset[i]) for i in topname_epis]
costsum_epis
#plt.scatter(sedsum_epis,costsum_epis, c = 'c', marker = 'o', label='bcr_ranking_epistasis')

sedsum_ignore_epis = [sum(sed_epis * dataset[i]) for i in topname]
sedsum_ignore_epis
costsum_ignore_epis = [sum(cost * dataset[i]) for i in topname]
costsum_ignore_epis
plt.scatter(sedsum_ignore_epis, costsum_ignore_epis,
            c='b', marker='o', label='ignore_epistas_bcr')
plt.scatter(sedsum_epis, costsum_epis, c='c',
            marker='o', label='epistasis_bcr')
#plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.grid(None)
plt.savefig('bcr_epistasis_maple29.png', dpi=300, bbox_inches='tight')
plt.show()


#plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')
#plt.scatter(sedsum_ignore_epis, costsum_ignore_epis, c='b', marker='x', label='bcr_ignore_epistasis')
plt.scatter(sedsum_epis, costsum_epis, c='c',
            marker='o', label='with_epistasis')
plt.scatter(sed_noepis, cost_noepis, c='m', marker='D', label='no_epistasis')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.grid(None)
#plt.savefig('noepistasis_maple29.png',dpi = 300,bbox_inches='tight')

plt.show()

#plt.scatter(sedsum_ignore_epis,costsum_ignore_epis, c = 'b', marker = 'o', label='bcr_ranking_epistasis')


#plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')
plt.scatter(sedsum_ignore_epis, costsum_ignore_epis, c='b',
            marker='x', label='bcr_ignore_epistasis')
plt.scatter(sedsum_epis, costsum_epis, c='c',
            marker='o', label='True_pf_epistasis')
#plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.grid(None)
#plt.savefig('bcrepis_maple29.png',dpi = 300,bbox_inches='tight')


plt.show()


# EA
parcelid = dataset.Site_ID
sed = dataset.SedRed
cost = dataset.Cost
clusters = dataset.NEAR_FID
clusterids = list(dataset.NEAR_FID.unique())
clusterids
clustersize = dataset.Cluster_size
sed
#clustersize = dataset.Cluster_size
sedrank = dataset.Sedrank
disrank = dataset.NEARrank

creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

# Create random items and store them in the items' dictionary.

items = {}
for i in range(NBR_ITEMS):
    items[i] = (cost[i], sed[i], clustersize[i], disrank[i])

items.values()


def initIndividual(icls, content):
    return icls(content)


def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)


IND_INIT_SIZE = 500
toolbox = base.Toolbox()
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, IND_INIT_SIZE)

ind2 = toolbox.individual()
ind2

toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list,
                 toolbox.individual_guess, data_folder/"bcrseeds_maple29.json")
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop1 = toolbox.population(10)
pop1

pop2 = toolbox.population_guess()
pop2


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
    CXPB = 0.9
    MUTPB = 0.1
    # no seeding
    pop = toolbox.population(n=MU)
    # seeding!!
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

# no seeding
noseedpop = results[0]
noseedstats = results[1]
noseedhof = results[2]
noseedhof
noseedfront = np.array([ind.fitness.values for ind in noseedpop])
noseedfront
noseedcost_f = noseedfront[:, 0]
noseedsed_f = noseedfront[:, 1]


###############
# bcr seeding
def main():
    NGEN = 1000
    MU = 100
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    # no seeding
    #pop = toolbox.population(n=MU)
    # seeding!!
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
bcrseed_front = np.array([ind.fitness.values for ind in bcrseed_pop])
bcrseed_front
bcrseedcost_f = bcrseed_front[:, 0]
bcrseedsed_f = bcrseed_front[:, 1]


plt.scatter(noseedsed_f, noseedcost_f, c='y', marker='v', label='EA_noseed')
plt.scatter(bcrseedsed_f, bcrseedcost_f, c='r', marker='s', label='EA_bcrseed')
plt.scatter(sedsum_ignore_epis, costsum_ignore_epis, c='b',
            marker='x', label='bcr_ignore_epistasis')

#plt.scatter(sedsum_epis, costsum_epis, c='c',
            #marker='o', label='bcr_consider_epistasis')
#plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')

plt.scatter(sedsum_epis, costsum_epis, c='c',
            marker='o', label='TruePF_epistasis')
#plt.scatter(sed_noepis ,cost_noepis, c = 'm', marker = 'D', label='no_epistasis_bcr')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.grid(None)
plt.savefig('EAbcr_epistasis.png', dpi=300, bbox_inches='tight')
#plt.savefig('seedvsnoseed_epistasis.png',dpi = 300,bbox_inches='tight')
#plt.savefig('bcr_epistasis.png',dpi = 300,bbox_inches='tight')
plt.show()

# Create the data.frame for performance evaluation NGEN = 1000

dic_EAseed = {'f1':bcrseedsed_f , 'f2':bcrseedcost_f.tolist()}
df_EAseed = pd.DataFrame(dic_EAseed) 
df_EAseed['prob'] = 'maple29'
df_EAseed['repl'] = 1
df_EAseed['algorithm'] = 'EA_bcrseeds'

dic_noseed = {'f1':noseedsed_f , 'f2':noseedcost_f.tolist()}
df_noseed = pd.DataFrame(dic_noseed) 
df_noseed['prob'] = 'maple29'
df_noseed['repl'] = 1
df_noseed['algorithm'] = 'EA_noseeds'

dic_bcr = {'f1':sedsum_ignore_epis, 'f2':costsum_ignore_epis}
df_bcr = pd.DataFrame(dic_bcr) 
df_bcr['prob'] = 'maple29'
df_bcr['repl'] = 1
df_bcr['algorithm'] = 'BCR_rank'

df_algorithm = pd.concat([df_EAseed,df_noseed, df_bcr])
df_algorithm
df_algorithm.to_csv(data_folder/'output/maple29_performance_1000GEN.csv', index = False)
