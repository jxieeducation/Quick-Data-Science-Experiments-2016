require(MASS)
data(iris)

#relatively_pure_example = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 2)
#unpure_example = c(1, 2, 2, 1, 2)
getGiniImpurity <- function(y){
  n = unique(y)
  sum_impurity = 0
  for(i in range(n)){
    f = sum(y == i) / length(y)
    sum_impurity = sum_impurity + f * (1 - f)
  }
  return(sum_impurity)
}
#print(getGiniImpurity(relatively_pure_example))
#print(getGiniImpurity(unpure_example))
summary(iris)

# first split, I am lazy atm, so just going to hard code
iris$class_s = (iris$Species == "setosa") * 1
plot(iris$Sepal.Length, iris$class_s)

split = min(iris$Sepal.Length) + 0.05
min_split = 999999
min_gini = 999999
while(split < max(iris$Sepal.Length)) {
  gini1 = getGiniImpurity(iris[iris$Sepal.Length < split,][,"class_s"])
  gini2 = getGiniImpurity(iris[iris$Sepal.Length >= split,][,"class_s"])
  totalGini = gini1 + gini2
  if(totalGini < min_gini){
    min_split <- split
    min_gini <- totalGini
  }
  split = split + 0.05
}
print(min_split)
print(min_gini)
