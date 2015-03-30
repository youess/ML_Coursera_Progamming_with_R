#!/usr/bin/env Rscript


randInitializeWeights <- function(l.in, l.out){

    epsilon.init <- .12

    w <- matrix(runif(l.out * (l.in + 1)), l.out, l.in + 1)

    w <- w * 2 * epsilon.init - epsilon.init

    w

}

