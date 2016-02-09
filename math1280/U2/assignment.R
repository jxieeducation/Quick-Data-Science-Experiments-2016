setwd("/Users/jason/Desktop/Quick-Data-Science-Experiments-2016/math1280/data/")

flower.data <- read.csv('flowers.csv')

x <- round(flower.data$Petal.Length, 0)
freq <- table(x)

names(flower.data)
nrow(flower.data)

x <- round(flower.data$Sepal.Length, 0)
freq <- table(x)

table(flower.data$Sepal.Width)
plot(table(flower.data$Sepal.Width))
cumsum(table(flower.data$Sepal.Width))

table(flower.data$Species)
