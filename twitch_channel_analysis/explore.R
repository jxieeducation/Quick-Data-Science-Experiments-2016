wings <- read.csv("/home/jason/Downloads/twitch_dataset/wings_sessions.csv", header=TRUE)
head(wings,10)
wings <- wings[order(wings$accessed_at_utc_std_max),]
head(wings,10)
small_wings <- wings[,c('viewers_max', 'viewers_min', 'viewers_sum', 'accessed_at_utc_std_max')]
head(small_wings, 10)
tail(small_wings, 10)

small_wings$date <-as.Date(small_wings$accessed_at_utc_std_max)
small_wings <- small_wings[,c('viewers_max', 'viewers_min', 'viewers_sum', 'date')]
head(small_wings,10)
unique(small_wings$date)

library(lattice)
library(chron)
summary(small_wings)
nrow(small_wings)
source("http://blog.revolutionanalytics.com/downloads/calendarHeat.R")
calendarHeat(small_wings$date, small_wings$viewers_max, varname="Calendar of SALTY WINGZ")





