
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rstratx

<!-- badges: start -->

<!-- badges: end -->

The **rstratx** provides an interface to
[stratx](https://github.com/parrt/stratx), a Python library for [A
Stratification Approach to Partial Dependence for Codependent
Variables](https://arxiv.org/abs/1907.06698). Currently, only the
StratPD algorithm is supported (which only applies to numeric features).

**WARNING:** This package is under heavy development. The underlying
Python code needs cleaned up, and imports aren’t really handled that
gracefully on the R side. Use at your own risk.

## Installation

``` r
# You can install the development version from GitHub:
if (!("remotes" %in% installed.packages()[, "Package"])) {
  install.packages("remotes")
}
remotes::install_github("bgreenwell/rstratx")
```

## Example

Here’s a basic example using the well-known Boston housing data set:

``` r
# Load required packages
library(pdp)      # for ordinary partial dependence
library(ranger)   # for random forest algorithm
#> Warning: package 'ranger' was built under R version 3.5.2
library(reticulate)  # for interfacing with Python
#> Warning: package 'reticulate' was built under R version 3.5.2
use_python("/Users/b780620/anaconda3/bin/python3", required = TRUE)  # FIXME
library(rstratx)  # for stratified partial dependence

# Load the Boston housing data
data(boston, package = "pdp")

#
# Ordinary partial dependence
#

# Fit a (default) random forest model and construct PDP for age
set.seed(1818)  # for reproducibility
rfo <- ranger(cmedv ~ ., data = boston)
partial(rfo, pred.var = "age", plot = TRUE)
```

<img src="man/figures/README-example-1.png" width="70%" />

``` r

#
# Stratified partial dependence
#

# Compute stratified partial dependence for age (auto fits an RF)
spd <- stratpd(
  X = subset(boston, select = -cmedv), 
  y = boston[, "cmedv", drop = FALSE],  # needs a one-column data frame (for now)
  feature_name = "age"
)

# Plot results
par(mar = c(4, 4, 1, 1) + 0.1)
plot(spd, type = "l", lwd = 2, las = 1, ylim = c(-10, 10))
```

<img src="man/figures/README-example-2.png" width="70%" />
