#!/usr/bin/env Rscript

nnCostFun <- function(nnPara, inputLayerSize,
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
    J <- 0

    K <- numLabel
    Y <- diag(K)[y, ]   # 5000 * 10

    # forward
    a1 <- cbind(1, X)   # 5000 * 401
    a2 <- cbind(1, sigmoid(a1 %*% t(theta1)))  # 26 * 5000
    h <- sigmoid(theta2 %*% t(a2))   # 10 * 5000

    costPos <- -Y * log(t(h))
    costNeg <- (1 - Y) * t(log(1 - h))
    costs <- costPos - costNeg

    J <- (1 / m) * sum(costs)

    # regularization
    theta1.filter <- theta1[, -1]
    theta2.filter <- theta2[, -1]
    reg <- (lambda / (2*m)) * (sum(theta1.filter^2) + sum(theta2.filter^2))
    J <- J + as.vector(reg)
    J
}
