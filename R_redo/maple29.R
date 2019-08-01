library(plyr)
library(reshape2)
library(ecr)
library(tidyverse)
library(ggplot2)
library(doBy)
rm(list = ls())
# reproducability
set.seed(42)
# setwd("C:/Users/langzx/Desktop/github/ESS521project/data")
# dataset <- read.csv('output/MAP29save_groupby.csv')
# ducksheet <- read.csv('output/wcmo_MAP_sub29.csv')
# duckdata <- ducksheet[c('ID', 'Duck')]
# duckdata
# dat <- left_join(dataset, duckdata, by = c("Site_ID" = "ID"))
# write.csv(x = dat,file = "dat_ECR.csv", row.names = FALSE)

dat <- read.csv(file = "dat_ECR.csv")

head(dat)
N <- dim(dat)[1]


top <- seq(10, 468, 5)
topname <- paste("Top", top, sep = "")

# Create new columns for bcr ranking
for (i in 1:length(top)) {
    dat[[topname[i]]] <- 0
    dat[[topname[i]]][which.maxn(dat$bcr, top[i])] <- 1
}
head(dat)

sed_noepis <- c(length = length(top))
cost_noepis <- c(length = length(top))
duck_noepis <- c(length = length(top))

for (i in 1: length(top)){
  sed_noepis[i] <- sum(dat$SedRed * dat[[topname[i]]])
  cost_noepis[i] <- sum(dat$Cost * dat[[topname[i]]])
  duck_noepis[i] <- sum(dat$Duck * dat[[topname[i]]])
}

##Save seeds
dat_seeds <- dat %>% dplyr:: select(grep("Top", names(dat)))
head(dat_seeds)
bcr_seeds_list <- as.list(dat_seeds[,1:length(top)])
bcr_seeds_list
### Add interaction based on distances to existent wetland

dat_epis <- dat %>% mutate (SedRed_epis = c(0*N))
dat_epis$SedRed_epis

for (i in 1:N) {
  dis_top <- round(dat_epis$Cluster_size[i]/2)
  dat_epis$SedRed_epis[i] <- ifelse (dat_epis$NEARrank[i] > dis_top, 0 ,  dat_epis$SedRed[i])
}
dat_epis <- dat_epis %>% mutate (bcr_epis = SedRed_epis/Cost)

top <- seq(10, 468, 5)
topname_epis <- paste("Top_epis", top, sep = "")

# Create new columns for bcr ranking
for (i in 1:length(top)) {
  dat_epis[[topname_epis[i]]] <- 0
  dat_epis[[topname_epis[i]]][which.maxn(dat_epis$bcr_epis, top[i])] <- 1
}
head(dat_epis)

sed_epis <- c(length = length(top))
cost_epis <- c(length = length(top))
duck_epis <- c(length = length(top))
sed_ignore_epis <- c(length = length(top))
cost_ignore_epis <- c(length = length(top))

for (i in 1: length(top)){
  sed_epis[i] <- sum(dat_epis$SedRed_epis * dat_epis[[topname_epis[i]]])
  cost_epis[i] <- sum(dat_epis$Cost * dat_epis[[topname_epis[i]]])
  sed_ignore_epis[i] <- sum(dat_epis$SedRed_epis * dat_epis[[topname[i]]])
  cost_ignore_epis[i] <- sum(dat_epis$Cost * dat_epis[[topname[i]]])
  #duck_epis[i] <- sum(dat$Duck * dat[[topname[i]]])
}

df_compare <- data.frame(cbind(sed_noepis,cost_noepis,sed_epis,cost_epis,sed_ignore_epis,cost_ignore_epis))
df_compare

ggplot(df_compare,aes(x, y = value, color = variable)) +
  geom_point(aes(x=sed_noepis,y=cost_noepis,col = "noepis"))+
  geom_point(aes(x=sed_epis,y=cost_epis,col = "withepis"))+
  geom_point(aes(x=sed_ignore_epis,y=cost_ignore_epis,col = "ignoreepis"))+


write.csv(x = dat_epis,file = "dat_epis.csv",row.names = FALSE)

## ECR

set.seed(42)

fitness.fun = function(x) {
  for (i in 1:N) {
    dis_top <- round(dat_epis$Cluster_size[i]/2)
    dat_epis$SedRed_epis_ecr[i] <- ifelse (dat_epis$NEARrank[i] > dis_top, 0 ,  dat_epis$SedRed[i])
  }
  sed = sum(x * dat_epis$SedRed_epis_ecr)
  cost = 1 * sum(x * dat$Cost)

  return (c(sed,cost))
}


n.bits = nrow(dat_epis)
#MU = 25L
LAMBDA = 10L
MAX.ITER = 1000L

ref.point = c(sum(dat_epis$SedRed_epis_ecr), sum(dat_epis$Cost))

# initialize toolbox
control = initECRControl(fitness.fun, n.objectives = 2L, minimize = c(FALSE,TRUE))
control = registerECROperator(control, "mutate", mutBitflip, p = 1 / n.bits)
control = registerECROperator(control, "recombine", recCrossover)
control = registerECROperator(control, "selectForMating", selSimple)
# NSGA2
control = registerECROperator(control, "selectForSurvival",selNondom)


# initialize population of MU random bitstring
#population = genBin(MU, n.bits)
population = initPopulation(mu = length(top), genBin, initial.solutions = bcr_seeds_list)
fitness = evaluateFitness(control, population)
fitness
population


for (i in seq_len(MAX.ITER)) {
  offspring = recombinate(control, population, fitness = fitness, lambda = LAMBDA, p.recomb = 0.8)
  fitness.o = evaluateFitness(control, offspring)
  #apply (MU + LAMBDA) selection
  sel = replaceMuPlusLambda(control, population, offspring, fitness, fitness.o)
  population = sel$population
  fitness = sel$fitness
}


print(population)
#print(population[[which.max(fitness)]])
print((fitness))

#check the number of objects selected
lapply(population,sum)

#FrontRes <- as.data.frame(t(fitness))

FrontRes_seeds <- as.data.frame(t(fitness))


#plot(FrontRes[,1],FrontRes[,2])
png("ParetoFronts.png", width = 10, height = 10, units = 'in', res = 300)

Plot1 <- ggplot() +

  geom_point(data = FrontRes, aes(x = FrontRes[,1], y = FrontRes[,2],col = "ECR"),
             size = 2) +
  geom_point(data = df_compare, aes(x = sed_epis, y = cost_epis, col = "True_epis"),
             size = 2) +
  geom_point(data = df_compare, aes(x=sed_ignore_epis,y=cost_ignore_epis,col = "BCR_ignore"),
             size = 2) +
  geom_point(data = FrontRes_seeds, aes(x = FrontRes_seeds[,1], y = FrontRes_seeds[,2],col = "ECR_seeds"),
             size = 2) +
  theme_minimal() +
  theme(legend.title=element_blank())+
  xlab(label = "SedRed") +
  ylab(label = "Cost") +
  ggtitle("Pareto fronts")
Plot1
dev.off()

# BLACK BOX implementation

# use "natural" binary representation x = (x_1, ..., x_n) with x_i = 1 means,
# that object i is bagged and x_i = 0 otherwise.
res = ecr(fitness.fun, n.objectives = 2L, minimize = c(FALSE,TRUE),
          representation = "binary", n.bits = nrow(dat_epis),
          mu = length(top), lambda = 10L, survival.strategy = "plus",
          initial.solutions = bcr_seeds_list,
          terminators = list(stopOnIters(1000L)))

# extract EA knapsack solution
res$pareto.front
print(res$pareto.set)


