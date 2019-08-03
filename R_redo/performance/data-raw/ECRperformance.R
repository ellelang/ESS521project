
#devtools::install_github("jakobbossek/ecr2")
rm(list = ls())
library(ecr)
library(dplyr)
library(ggplot2)
data(mcMST)
mcMST
saveRDS(mcMST, "mcMST.rds")
head(mcMST)
obj.col <- c("f1", "f2")
mcMST <- dplyr::filter(mcMST,
                          grepl("100", prob), grepl("NSGA", algorithm))

write.csv(x = mcMST, file = "example_mcMST.csv", row.names = FALSE)
dim(mcMST)
mcMST <- ecr::normalize(mcMST, obj.cols = obj.col, offset = 1)
head(mcMST)
mcMST
ranks <- ecr::computeDominanceRanking(
  mcMST, obj.cols = obj.col
)

ecr::plotDistribution(ranks) +
  ggplot2::theme(legend.position = "none")

ecr::plotScatter2d(
  dplyr::filter(mcMST, repl <= 2),
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
  mcMST, unary.inds = unary.inds
)

head(inds$unary)

ecr::plotDistribution(inds$unary, plot.type = "boxplot")
toLatex(inds$unary, stat.cols = c("HV", "EPS"))

unary <- inds$unary
unary$algorithm <- gsub(
  "NSGA2.", "",
  unary$algorithm, fixed = TRUE)

tests <- test(
  unary,
  col.names = c("HV", "EPS", "ONVG")
)
ecr::TEST