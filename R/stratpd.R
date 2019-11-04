#' Stratified partial dependence
#'
#' TBD.
#'
#' @param X A matrix-like R object (e.g., a data frame or matrix) containing
#' ONLY the feature columns from the training data.
#'
#' @param y A vector or matrix-like R object (e.g., a data frame or matrix)
#' containing ONLY the target values from the training data.
#'
#' @param feature_name Character string specifying the feature of interest.
#'
#' @param ... Additional optional arguments.
#'
#' @export
stratpd <- function(X, y, feature_name = NULL, ...) {
  if (!is.data.frame(X)) {
    X <- as.data.frame(X)
  }
  if (!is.data.frame(y)) {
    y <- as.data.frame(y)
  }
  if (is.null(feature_name)) {
    feature_name <- colnames(X)[1L]
  }
  stratx <- reticulate::import_from_path(
    module = "stratx",
    path = system.file("python", "stratx", package = "rstratx"),
    convert = TRUE
  )
  spd <- stratx$partdep$plot_stratpd(
    X = X,
    y = y,
    colname = feature_name,
    targetname = colnames(y),
    ...
  )
  names(spd) <- c("leaf_xranges", "leaf_slopes", "pdpx", "pdpy", "ignored")
  class(spd) <- c("stratpd", class(spd))
  attr(spd, which = "feature_name") <- feature_name
  attr(spd, which = "target_name") <- names(y)
  spd
}


#' Plot stratified partial dependence functions
#'
#' Extends R's generic \code{\link[graphics]{plot}} function to handle objects
#' that inherit from class \code{"stratpd"}.
#'
#' @param x AN object that inherits from class \code{"stratpd"}.
#'
#' @param ... Additional optional arguments to be passed onto
#' \code{\link[graphics]{plot}}.
#'
#' @rdname plot.stratpd
#'
#' @export
plot.stratpd <- function(x, ...) {
  plot(
    x = x[["pdpx"]],
    y = x[["pdpy"]],
    xlab = attr(x, which = "feature_name"),
    ylab = attr(x, which = "target_name"),
    ...
  )
  for (i in seq_len(nrow(x[["leaf_xranges"]]))) {
    w <- diff(x[["leaf_xranges"]][i, ])
    delta_y <- x[["leaf_slopes"]][i] * w
    closest_x_i <- which.min(abs(x[["pdpx"]] - x[["leaf_xranges"]][i, 1L]))
    closest_x <- x[["pdpx"]][closest_x_i]
    closest_y <- x[["pdpy"]][closest_x_i]
    segments(
      x0 = closest_x,
      y0 = closest_y,
      x1 = closest_x + w,
      y1 = closest_y + delta_y,
      adjustcolor("red", alpha.f = 0.3)
    )
  }
}
