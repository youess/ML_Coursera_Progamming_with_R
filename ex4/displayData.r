#!/usr/bin/env Rscript

# rotate a matrix 90 degrees by clockwise
rotate <- function(x) t(apply(x, 2, rev))

pixelMatrix <- function(x) rotate(matrix(x/max(abs(x)), 20, 20))

# image(pixelMatrix(X[2100, ]), col=gray.colors(100), axes=F)

displayData <- function(x, ...){
    if (nrow(x) == 1) {
        op <- par(mar=c(.1, .1, 2, .1))
        image(pixelMatrix(x), col=gray.colors(100),
              axes=F, ...)
    }
    else{
        op <- par(mar=c(.1, .1, .1, .1))
        width <- floor(sqrt(nrow(x)))
        height <- ceiling(nrow(x) / width)
        layout(matrix(c(1:nrow(x), rep(0, abs(width*height - nrow(x)))),
                    height, width))
        for (i in 1:nrow(x)){
            image(pixelMatrix(x[i, ]), col=gray.colors(100), axes=F,
                ...)
        }
    }
    par(op)
}

