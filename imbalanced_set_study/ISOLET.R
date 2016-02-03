isolet <- read.csv(file="/Users/jason/Downloads/isolet1+2+3+4.data", sep=",")
colnames(isolet) <- paste0("f", as.character(1:618))
isolet$f618 <- (isolet$f618 == 13) * 1

library(caret)

# this is normal
train_ind <- sample(nrow(isolet) * 0.7)
isolet_train <- isolet[train_ind,]
isolet_test <- isolet[-train_ind,]
normal_model <- glm(f618~., family = binomial(link=logit), data=isolet_train)
summary(normal_model)
predictions <- (predict(normal_model, isolet_test, type="response") > 0.5) * 1
res <- confusionMatrix(predictions, isolet_test$f618)
res$table
res$overall # 9.893162e-01, 67 / 72 recall

# this is balanced
positive <- isolet[isolet$f618 == 1,]
negative <- isolet[isolet$f618 == 0,]
balanced_isolet <- rbind(positive, negative[sample(nrow(positive)),])
balanced_model <- glm(f618~., family = binomial(link=logit), data=balanced_isolet)
predictions <- (predict(balanced_model, isolet_test, type="response") > 0.5) * 1
res <- confusionMatrix(predictions, isolet_test$f618)
res$table
res$overall #4.855769e-01, 100% recall

# this is more balanced
positive <- isolet[isolet$f618 == 1,]
negative <- isolet[isolet$f618 == 0,]
mbalanced_isolet <- rbind(positive, negative[sample(nrow(positive) * 3),])
mbalanced_model <- glm(f618~., family = binomial(link=logit), data=mbalanced_isolet)
predictions <- (predict(mbalanced_model, isolet_test, type="response") > 0.5) * 1
res <- confusionMatrix(predictions, isolet_test$f618)
res$table
res$overall #8.947650e-01, 100% recall

# this is more more balanced
positive <- isolet[isolet$f618 == 1,]
negative <- isolet[isolet$f618 == 0,]
mmbalanced_isolet <- rbind(positive, negative[sample(nrow(positive) * 5),])
mmbalanced_model <- glm(f618~., family = binomial(link=logit), data=mmbalanced_isolet)
predictions <- (predict(mmbalanced_model, isolet_test, type="response") > 0.5) * 1
res <- confusionMatrix(predictions, isolet_test$f618)
res$table
res$overall #9.220085e-01, 100% recall

#### 1 to 5 seems to be a good ratio, gonna try bagging
library(foreach)
iterations<-50
predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
  positive <- isolet_train[isolet$f618 == 1,]
  negative <- isolet_train[isolet$f618 == 0,]
  p_num <- round(nrow(positive) * 1)
  n_num <- round(nrow(positive) * 5)
  t_isolet <- rbind(positive[sample(1:nrow(positive), p_num),], negative[sample(1:nrow(negative), n_num),])
  model <- glm(f618~., family = binomial(link=logit), data=t_isolet)
  predict(model, isolet_test, type="response")
}
predictions<-rowMeans(predictions)
predictions <- (predictions > 0.5) * 1
res <- confusionMatrix(predictions, isolet_test$f618)
res$table
res$overall

#1,97.4,48
#3,98.2,50
#5,98.3,53
#10,98.6,55
#50,


