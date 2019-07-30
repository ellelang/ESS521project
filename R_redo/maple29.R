library(plyr)
library(reshape2)
library(ecr)
library(tidyverse)
rm(list=ls())
# reproducability
set.seed(42)
setwd("C:/Users/langzx/Desktop/github/ESS521project/data")
dataset <- read.csv('output/MAP29save_groupby.csv')
ducksheet <- read.csv('output/wcmo_MAP_sub29.csv')
duckdata <- ducksheet[c('ID','Duck')]
duckdata
dat <- left_join(dataset,duckdata, by = c("Site_ID"="ID") )
head(dat)
