library("e1071")
attach(iris)

x <- subset(iris, select=-Species)
y <- Species

svm_model <- svm(Species ~ Petal.Width + Petal.Length, data=iris, kernel="linear")
summary(svm_model)
pred <- predict(svm_model, x)
table(pred,y)
plot(svm_model, iris, Petal.Width ~ Petal.Length)

svm_model <- svm(Species ~ Petal.Width + Petal.Length, data=iris, kernel="polynomial")
summary(svm_model)
pred <- predict(svm_model, x)
table(pred,y)
plot(svm_model, iris, Petal.Width ~ Petal.Length, slice = list(Sepal.Width = 3, Sepal.Length = 4))

svm_model <- svm(Species ~  Petal.Width + Petal.Length, data=iris, kernel="radial")
summary(svm_model)
pred <- predict(svm_model, x)
table(pred,y)
plot(svm_model, iris, Petal.Width ~ Petal.Length, slice = list(Sepal.Width = 3, Sepal.Length = 4))

svm_model <- svm(Species ~  Petal.Width + Petal.Length, data=iris, kernel="sigmoid")
summary(svm_model)
pred <- predict(svm_model, x)
table(pred,y)
plot(svm_model, iris, Petal.Width ~ Petal.Length, slice = list(Sepal.Width = 3, Sepal.Length = 4))

