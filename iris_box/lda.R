require(MASS)
data(iris)
head(iris, 3)

r <- lda(formula = Species ~ ., 
         data = iris, 
         prior = c(1,1,1)/3)
r$prior
r$counts
r$means
r$scaling
r$svd

prop = r$svd^2/sum(r$svd^2)
