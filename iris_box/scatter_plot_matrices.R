require(MASS)
data(iris)


########## method 1: cov
data(iris)
cor(iris[,1:4]) # covariance is a measure of how one variable is associated with another

########## method 2: scatter plot matrix
cols <- character(nrow(iris))
cols[] <- "black"

cols[iris$Species == "versicolor"] <- "blue"
cols[iris$Species == "setosa"] <- "green"
cols[iris$Species == "virginica"] <- "red"
pairs(iris[,1:4],col=cols, labels=c("s. length", "s. width", "c. length", "c. width"))
