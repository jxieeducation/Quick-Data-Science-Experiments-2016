library("e1071")

head(iris,5)
attach(iris)

x <- subset(iris, select=-Species)
y <- Species
svm_model <- svm(Species ~ ., data=iris)
summary(svm_model)
pred <- predict(svm_model, x)
table(pred,y)

svm_tune <- tune(svm, train.x=x, train.y=y, kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)
svm_model_after_tune <- svm(Species ~ ., data=iris, kernel="radial", cost=1, gamma=0.5)
table(pred,y)
