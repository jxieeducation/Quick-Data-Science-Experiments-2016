library(dplyr)
library(kohonen)

data(iris)
iris

iris_data <- select(iris, Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
kohmap <- som(scale(iris_data), grid = somgrid(5, 5, "hexagonal"))


