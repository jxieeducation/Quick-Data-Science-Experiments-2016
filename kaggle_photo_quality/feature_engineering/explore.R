library(dplyr)

setwd("/Users/jason.xie/Downloads/Quick-Data-Science-Experiments-2016/kaggle_photo_quality")

list.files("data/")
train <- read.csv("data/training.csv", stringsAsFactors=FALSE)
test <- read.csv("data/test.csv", stringsAsFactors=FALSE)

summary(train)
summary(test)

good <- filter(train, good == 1)
bad <- filter(train, good == 0)
nrow(good)
nrow(bad) # there are 3x the bad than the good
summary(good)
summary(bad)

cor(train$size, train$good) # 10% pearson correlation for size
cor(train$width, train$good) # 6% for width
cor(train$height, train$good) # 0.3% for height, so width is more important than height???
cor(train$latitude, train$good) # -10% for lat, how vertical something is
cor(train$longitude, train$good) # 20% for long, how horizontal something is o.0 wtf

# longitude and latitude map http://www.mapsofworld.com/images2008/world-map-with-latitude-and-longitude.jpg
summary(good$latitude)
summary(bad$latitude) # the mean latitude of good is 32, while bad is 27

getLength <- function(line){
  return(length(unlist(strsplit(line, "[ ]"))))
}
train$name_length <- sapply(train$name, FUN=getLength)
cor(train$name_length, train$good) # well guess name length is useless
train$description_length <- sapply(train$description, FUN=getLength)
cor(train$description_length, train$good) # 1% correlation, not strong...




