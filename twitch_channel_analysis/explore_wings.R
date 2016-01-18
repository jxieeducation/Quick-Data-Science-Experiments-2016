wings <- read.csv("/Users/jason/Downloads/twitch-info/wings_sessions.csv", header=TRUE)
wings <- wings[order(wings$accessed_at_utc_std_max),]
small_wings <- wings[,c('viewers_max', 'viewers_min', 'viewers_sum', 'accessed_at_utc_std_max', 'accessed_at_utc_std_min')]

small_wings$dateMax <- as.POSIXlt(small_wings$accessed_at_utc_std_max, format="%Y-%m-%d %H:%M:%S")
small_wings$dateMin <- as.POSIXlt(small_wings$accessed_at_utc_std_min, format="%Y-%m-%d %H:%M:%S")
small_wings <- small_wings[,c('viewers_max', 'viewers_min', 'viewers_sum', 'dateMax', 'dateMin')]
small_wings$duration <- difftime(small_wings$dateMax, small_wings$dateMin, units=c("mins"))

library(lattice)
library(chron)
source("http://blog.revolutionanalytics.com/downloads/calendarHeat.R")
calendarHeat(small_wings$dateMin, small_wings$viewers_max, varname="Viewership Heatmap")
calendarHeat(small_wings$dateMin, small_wings$duration, varname="Stream Duration Heatmap")

# is there a corrolation between viewership and stream duration
ggplot(small_wings, aes(x=as.numeric(duration), y=viewers_min)) + geom_point() + ggtitle("Corrolation between duration and min viewership") # LOL
ggplot(small_wings, aes(x=as.numeric(duration), y=viewers_max)) + geom_point() + ggtitle("Corrolation between duration and max viewership") + theme(text = element_text(size=12))
 # show this info graphics
ggplot(small_wings, aes(x=as.numeric(duration), y=viewers_sum)) + geom_point() + ggtitle("Corrolation between duration and sum viewership")


library(ggplot2)
library(reshape2)
wings_long <- melt(small_wings[,c('viewers_max', 'viewers_min', 'dateMin')], id="dateMin")
wings_long <- wings_long[1:72,]
ggplot(data=wings_long, aes(x=date, y=value, colour=variable)) + geom_line() + ggtitle("Wings viewership")

