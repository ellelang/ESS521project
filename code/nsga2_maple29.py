import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

parcelid =  dataset.Site_ID
sed = dataset.SedRed
cost = dataset.Cost
clusters = dataset.NEAR_FID
clusterids = list(dataset.NEAR_FID.unique())
clusterids
sed
clustersize = dataset.Cluster_size
sedrank = dataset.Sedrank
disrank = dataset.NEARrank

creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

