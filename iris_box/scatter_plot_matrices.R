require(MASS)
data(iris)


########## method 1: cov
data(iris)
cor(iris[,1:4]) # covariance is a measure of how one variable is associated with another

########## method 2: scatter plot matrix
pairs(iris[,1:4])
