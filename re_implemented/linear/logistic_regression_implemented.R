require(MASS)
data(iris)

plot(iris[,1:4])
cor(iris[,1:4])

X = iris$Sepal.Length
y = iris$Species
transformY <- function(t){
  n = 0
  if(t == "setosa"){
    n = 1
  }
  return (n)
}
y = sapply(y, transformY)

# now we start coding the logistic regression stuff
# log(OR) = a + b * X = k
# p(x) = e^k / (e^k + 1)

a = 2
b = 3

prob <- function(a, b, X){
  o = a + b * X
  return(e^k / (o + 1))
}
# this is the logloss function
cost <- function(a, b, X, y){
  mle = 0
  for(i in 1:length(X)){
    prob_t = prob(a, b, X[i])
    if(y[i] == 1){
      mle = mle + log(prob_t)
    } else {
      mle = mle + log(1 - prob_t)
    }
  }
  return(-mle/length(X))
}
cost(a, b, X, y)

grad_







