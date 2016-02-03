red <- read.csv(file="/Users/jason/Desktop/Quick-Data-Science-Experiments-2016/imbalanced_set_study/winequality-red.csv", head=TRUE, sep=";")
white <- read.csv(file="/Users/jason/Desktop/Quick-Data-Science-Experiments-2016/imbalanced_set_study/winequality-white.csv", head=TRUE, sep=";")
red$identity <- rep(1, nrow(red))
white$identity <- rep(0, nrow(white))
red <- red[sample(1:nrow(red), 250, replace = FALSE),]
wine <- rbind(red, white)
train_ind <- sample(nrow(wine) * 0.9)
wine_train <- wine[train_ind,]
wine_test <- wine[-train_ind,]

library(caret)
normal_model <- glm(identity~., family = binomial(link=logit), data=wine_train)
summary(normal_model)
predictions <- (predict(normal_model, wine_test, type="response") > 0.5) * 1
confusionMatrix(predictions, wine_test$identity)
