require(MASS)
data("Boston")

summary(Boston)
plot(density(Boston$medv))

y <- Boston$medv
x <- Boston$rm

x <- cbind(1,x)
theta<- c(0,0)
m <- nrow(x)

costFunction <- function(x, y, theta){
  cost <- sum(((x%*%theta)- y)^2)/(2*m)
  return(cost)
}

cost <- costFunction(x, y, theta)

alpha <- 0.001
iterations <- 1500

prev_cost <- 9999
for(i in 1:iterations)
{
  theta[1] <- theta[1] - alpha * (1/m) * sum(((x%*%theta)- y))
  theta[2] <- theta[2] - alpha * (1/m) * sum(((x%*%theta)- y)*x[,2])
  cost <- costFunction(x, y, theta)
  print(cost)
  if(abs(prev_cost - cost) < 0.0001){
    break
  }
  prev_cost <- cost
}

predicted <- x%*%theta
residual <- y - predicted
plot(residual)
