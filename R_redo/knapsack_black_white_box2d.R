library(plyr)
library(reshape2)
library(ecr)


rm(list=ls())

# reproducability
set.seed(1)

# SOLVING THE 0-1-KNAPSACK PROBLEM WITH A GENETIC ALGORITHM
# =========================================================
# Given a knapsack with a weight capacity limit and a set of n objects with each
# a weight and a value, we aim to fill the knapsack with objects in a way that
# the capacity limit is respected and the sum of values is maximized.

# knapsack with 10 objects A, B, ..., J with weight w_i and value v_i
ks = data.frame(
  object = LETTERS[1:10],
  weight = c(10, 11, 4, 2, 2, 9, 3, 5, 15, 4),
  value1  = c(5, 10, 1, 6, 3, 5, 3, 19, 23, 1),
  value2 = c(20, 1, 10, 60, 1, 50, 3, 9, 2, 10)
)

# knapsack capacity
ks.limit = 25L

# objective function to be maximized, i.e., total value of bagged objects,
# under the restriction of the total weight being lower than the capacity.
fitness.fun = function(x) {
  val1 = sum(x * ks$value1)
  val2 = sum(x * ks$value2)
  weight = sum(x * ks$weight)
  if (weight > ks.limit)
    return(0) # penalty for limit violation
  return (c(val1,val2))
}

# use "natural" binary representation x = (x_1, ..., x_n) with x_i = 1 means,
# that object i is bagged and x_i = 0 otherwise.
res = ecr(fitness.fun, n.objectives = 2L, minimize = c(FALSE,FALSE),
          representation = "binary", n.bits = nrow(ks),
          mu = 25L, lambda = 10L, survival.strategy = "plus",
          terminators = list(stopOnIters(500L)))

# extract EA knapsack solution
print(res$pareto.front)
print(res$pareto.set)


pl.front = plotFront(res$pareto.front, obj.names = c("Val1", "Val2")) + ggtitle("")

print(pl.front)

# write as the "white box" sequence

# some errors still exist -- fixed Sept 4 with J Bossek suggestion

n.bits = nrow(ks)
MU = 25L
LAMBDA = 10L
MAX.ITER = 300L


# initialize toolbox
control = initECRControl(fitness.fun, n.objectives = 2L, minimize = c(FALSE, FALSE))
control = registerECROperator(control, "mutate", mutBitflip, p = 1 / n.bits)
control = registerECROperator(control, "recombine", recCrossover)
control = registerECROperator(control, "selectForMating", selSimple)
control = registerECROperator(control, "selectForSurvival", selNondom)

# initialize population of MU random bitstring
population = genBin(MU, n.bits)
fitness = evaluateFitness(control, population)


# now do the evolutionary loop
for (i in seq_len(MAX.ITER)) {

  offspring = recombinate(control, population, fitness = fitness, lambda = LAMBDA, p.recomb = 0.7)
  offspring = mutate(control, offspring, p.mut = 0.3)
  

  fitness.o = evaluateFitness(control, offspring)
  
  # apply (MU + LAMBDA) selection
  sel = replaceMuPlusLambda(control, population, offspring, fitness, fitness.o)
  population = sel$population
  fitness = sel$fitness
}



print(population[[which.max(fitness)]])
print(max(fitness))
