#!/usr/bin/env Rscript

nnGradFun <- function(nnPara, inputLayerSize,
                      hiddenLayerSize, numLabel,
                      X, y, lambda)
{
    # reshape weights of parameters into matrix
    theta1 <- array(nnPara[1:(hiddenLayerSize * (1 + inputLayerSize))],
                    c(hiddenLayerSize, 1 + inputLayerSize))   # 25 * 401
    theta2 <- array(nnPara[(1 + (hiddenLayerSize * (1 + inputLayerSize))):length(nnPara)],
                    c(numLabel, hiddenLayerSize + 1))         # 10 * 26
    # number of samples
    m <- nrow(X)

    K <- numLabel
    Y <- diag(K)[y, ]   # 5000 * 10

    # forward
    a1 <- cbind(1, X)   # 5000 * 401
    a2 <- cbind(1, sigmoid(a1 %*% t(theta1)))  # 26 * 5000
    h <- sigmoid(theta2 %*% t(a2))   # 10 * 5000

    theta1.filter <- theta1[, -1]
    theta2.filter <- theta2[, -1]

    # back propagation
    delta1 <- delta2 <- 0
    for (i in 1:m){
        # step 1
        a1 <- c(1, t(X[i, ]))
        z2 <- theta1 %*% a1
        a2 <- c(1, sigmoid(z2))
        z3 <- theta2 %*% a2
        a3 <- sigmoid(z3)

        # step 2
        yi <- Y[i, ]
        d3 <- a3 - yi

	    # step 3
	    d2 = (t(theta2.filter) %*% d3) * sigmoidGradient(z2)

        # step 4
	    delta2 <- delta2 + (d3 %*% t(a2))
        delta1 <- delta1 + (d2 %*% t(a1))

    }

    # step 5
    theta1.grad <- (1/m) * delta1
    theta2.grad <- (1/m) * delta2

    # add regularization
    theta1.grad[, -1] <- theta1.grad[, -1] + (lambda / m) * theta1.filter
    theta2.grad[, -1] <- theta2.grad[, -1] + (lambda / m) * theta2.filter

    # unroll gradients
    c(theta1.grad, theta2.grad)
}
