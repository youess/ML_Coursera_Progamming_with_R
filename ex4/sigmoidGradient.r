#!/usr/bin/env Rscript

sigmoidGradient <- function(z){

    gz <- sigmoid(z)
    gz * (1 - gz)

}
