# # Global reference(s) to Python module(s) (will be initialized in .onLoad)
# plt <- NULL
#
# .onLoad <- function(libname, pkgname) {
#   # Use superassignment to update global reference(s) to Python module(s)
#   plt <<- reticulate::import("matplotlib.pyplot", delay_load = TRUE)
# }
