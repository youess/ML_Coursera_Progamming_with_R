
# warm up exercise

## diagonal matrix

m <- diag(5)
diag(m) <- 10
m

# ex1, univariate regression

## load the data
dat <- read.table("./mlclass-ex1/ex1data1.txt", sep=",")

## plot the data
x <- dat[, 1]
y <- dat[, 2]

plot(x, y, col="red", pch=3,
     xlab="Population of City in 10,000s",
     ylab="Profit in $10,000s")

## univariate gradient

X <- cbind(1, x)
theta <- rep(0, 2)

iterations <- 1500
alpha <- 0.01

computeCost <- function(X, y, theta){
    n <- length(y)
    cost <- (1/(2 * n)) * t(X %*% theta - y) %*% (X %*% theta - y)
    as.vector(cost)
}

gradientDescent <- function(X, y, theta, alpha, iter.num){
    n <- length(y)
    cost.record <- rep(0, iter.num)
    for (i in 1:iter.num){
        # update regression parametrs
        h <- t(X %*% theta - y)
        m <- length(theta)
        tmp <- rep(0, m)
        for (j in 1:m){
            tmp[j] <- theta[j] - alpha * (1/n) * as.vector(h %*% X[, j])
        }
        theta <- tmp
        # record the cost history
        cost.record[i] <- computeCost(X, y, theta)
    }
    list(theta = theta, costs = cost.record)
}

res <- gradientDescent(X, y, theta, alpha, iterations)
model.theta <- res$theta

c(1, 3.5) %*% model.theta
c(1, 7) %*% model.theta

abline(model.theta[1], model.theta[2], col="blue")

theta0 <- seq(-10, 10, len=100)
theta1 <- seq(-1, 4, len=100)

cost.m <- matrix(0, 100, 100)
for (i in 1:100){
    for (j in 1:100){
        tmp <- c(theta0[i], theta1[j])
        cost.m[i, j] <- computeCost(X, y, tmp)
    }
}

require("colorRamps")
persp(theta0, theta1, cost.m, col=matlab.like(100))

contour(theta0, theta1, log10(cost.m), col=c(rev(blues9), "purple"))
points(model.theta[1], model.theta[2], pch=3, col="red", cex=2)

## multiple variables with linear regression


