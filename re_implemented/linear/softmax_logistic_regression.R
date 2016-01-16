require(MASS)
data(iris)
unique(iris$Species)

x <- iris
x$class_s <- 1 * (x$Species == "setosa")
x$class_ver <- 1* (x$Species == "versicolor")
x$class_vir <- 1 * (x$Species == "virginica")
x <- x[,c(1:4, 6:8)]
names(x)
summary(x)
plot(x)
plot(x$class_s)
plot(x$class_ver)
plot(x$class_vir)

s_model <- glm(class_s ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, family = binomial(logit), data=x)
ver_model <- glm(class_ver ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, family = binomial(logit), data=x)
vir_model <- glm(class_vir ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, family = binomial(logit), data=x)
summary(s_model)
s_pr <- predict(s_model, x, type="response")
round(s_pr, 2)
table(actual=x$class_s, predicted=s_pr>.5) # showing confusion matrix of classification

# softmax logic
x <- cbind(x,1)
s_model$coefficients
# predicting x row=51
row_num = 51
num_s = exp(sum(s_model$coefficients * x[row_num, c(8, 1:4)]))
num_ver = exp(sum(ver_model$coefficients * x[row_num, c(8, 1:4)]))
num_vir = exp(sum(vir_model$coefficients * x[row_num, c(8, 1:4)]))
denim = exp(sum(s_model$coefficients * x[row_num, c(8, 1:4)])) + exp(sum(ver_model$coefficients * x[row_num, c(8, 1:4)])) + exp(sum(vir_model$coefficients * x[row_num, c(8, 1:4)]))
print(paste0("chance of setosa: ", num_s / denim))
print(paste0("chance of ver: ", num_ver / denim))
print(paste0("chance of vir: ", num_vir / denim))
