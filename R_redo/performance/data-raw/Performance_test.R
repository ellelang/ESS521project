#devtools::install_github("jakobbossek/ecr2")
rm(list = ls())
library(ecr)
library(dplyr)
library(ggplot2)
#library(tibble)


setwd("C:/Users/langzx/Desktop/github/ESS521project/data")
#The installed ECR package didn't include the "test" function
source("test.R")
# data <- read.csv ('output/df_performance_0802.csv')
# data$prob <- type.convert(data$prob, as.is = TRUE) 
# data$algorithm <- type.convert(data$algorithm, as.is = TRUE) 
# 
# data <- as_tibble(data)
# data
# 
# saveRDS(data, "df_performace_0802.rds")
EA_rds <- readRDS("df_performace_0802.rds")
head(EA_rds)


obj.col <- c("f1", "f2")
dat_normal <- ecr::normalize(EA_rds , obj.cols = obj.col, offset = 1)
dat_normal
ranks <- ecr::computeDominanceRanking(
  dat_normal, obj.cols = obj.col
)

ecr::plotScatter2d(
  dplyr::filter(dat_normal, repl <= 2),
  facet.type = "grid",
  shape = "algorithm",
  facet.args = list(facets = formula(repl ~ prob))) +
  ggplot2::scale_color_grey(end = 0.8)


myONVG <- ecr::makeEMOAIndicator(
  fun = function(points, ...) ncol(points),
  name = "ONVG",
  latex.name = "I_{0}",
  minimize = FALSE
)

myONVG

unary.inds <- list(
  list(fun = ecr::emoaIndHV),
  list(fun = ecr::emoaIndEps),
  list(fun = myONVG)
)

inds <- ecr::computeIndicators(
  dat_normal, unary.inds = unary.inds
)

head(inds$unary)

ecr::plotDistribution(inds$unary, plot.type = "boxplot")
toLatex(inds$unary, stat.cols = c("HV", "EPS"))

unary <- inds$unary
unary$algorithm <- gsub(
  "NSGA2.", "",
  unary$algorithm, fixed = TRUE)
unary

tests <- test(
  unary,
  cols = c('HV','EPS','ONVG')
)

ecr::toLatex(tests)
