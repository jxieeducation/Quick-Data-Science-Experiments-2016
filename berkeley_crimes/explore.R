library(ggplot2)
library(dplyr)
library(ggmap)
library(stringr)

berkeley_crime <- read.csv("/Users/jason/Desktop/Quick-Data-Science-Experiments-2016/berkeley_crimes/Crime_Incidents.csv")

map.berkeley <- qmap("berkeley", zoom = 14, source="stamen", maptype="toner",darken = c(.3,"#BBBBBB"))
map.berkeley

berkeley_crime$Block_Location <- as.character(berkeley_crime$Block_Location)
berkeley_crime$location <- str_split_fixed(berkeley_crime$Block_Location, "\\n", 3)[,3]

berkeley_crime <- filter(berkeley_crime, location != "")

str_split_fixed(berkeley_crime$location, ", ", 2)[,1]
berkeley_crime$longitude <- as.numeric(substring(str_split_fixed(berkeley_crime$location, ", ", 2)[,1], 2))
berkeley_crime$latitude <- as.numeric(substring(str_split_fixed(berkeley_crime$location, ", ", 2)[,2], 1, 8))

map.berkeley + geom_point(data=berkeley_crime, aes(y=longitude, x=latitude), color="dark green", alpha=.03, size=1.1)

# overall map
map.berkeley +
  stat_density2d(data=berkeley_crime, aes(x=latitude, y=longitude, color=..density.., size=ifelse(..density..<=1,0,..density..), alpha=..density..), geom="tile",contour=F) +
  scale_color_continuous(low="orange", high="red", guide = "none") +
  scale_size_continuous(range = c(0, 3), guide = "none") +
  scale_alpha(range = c(0,.5), guide="none") +
  ggtitle("Berkeley Crime Heatmap") +
  theme(plot.title = element_text(family="Trebuchet MS", size=36, face="bold", hjust=0, color="#777777"))

levels(berkeley_crime$OFFENSE)
table(berkeley_crime$OFFENSE)

map.berkeley +
  stat_density2d(data=filter(berkeley_crime, OFFENSE == "BURGLARY AUTO"), aes(x=latitude, y=longitude, color=..density.., size=ifelse(..density..<=1,0,..density..), alpha=..density..), geom="tile",contour=F) +
  scale_color_continuous(low="orange", high="red", guide = "none") +
  scale_size_continuous(range = c(0, 3), guide = "none") +
  scale_alpha(range = c(0,.5), guide="none") +
  ggtitle("Berkeley Car Theft Heatmap") +
  theme(plot.title = element_text(family="Trebuchet MS", size=36, face="bold", hjust=0, color="#777777"))

map.berkeley +
  stat_density2d(data=filter(berkeley_crime, OFFENSE == "DISTURBANCE"), aes(x=latitude, y=longitude, color=..density.., size=ifelse(..density..<=1,0,..density..), alpha=..density..), geom="tile",contour=F) +
  scale_color_continuous(low="orange", high="red", guide = "none") +
  scale_size_continuous(range = c(0, 3), guide = "none") +
  scale_alpha(range = c(0,.5), guide="none") +
  ggtitle("Berkeley Disturbance Heatmap") +
  theme(plot.title = element_text(family="Trebuchet MS", size=36, face="bold", hjust=0, color="#777777"))

map.berkeley +
  stat_density2d(data=filter(berkeley_crime, OFFENSE == "DOMESTIC VIOLENCE"), aes(x=latitude, y=longitude, color=..density.., size=ifelse(..density..<=1,0,..density..), alpha=..density..), geom="tile",contour=F) +
  scale_color_continuous(low="orange", high="red", guide = "none") +
  scale_size_continuous(range = c(0, 3), guide = "none") +
  scale_alpha(range = c(0,.5), guide="none") +
  ggtitle("Berkeley Domestic Violence Heatmap") +
  theme(plot.title = element_text(family="Trebuchet MS", size=36, face="bold", hjust=0, color="#777777"))

map.berkeley +
  stat_density2d(data=filter(berkeley_crime, OFFENSE == "ASSAULT/BATTERY MISD."), aes(x=latitude, y=longitude, color=..density.., size=ifelse(..density..<=1,0,..density..), alpha=..density..), geom="tile",contour=F) +
  scale_color_continuous(low="orange", high="red", guide = "none") +
  scale_size_continuous(range = c(0, 3), guide = "none") +
  scale_alpha(range = c(0,.5), guide="none") +
  ggtitle("Berkeley Assult Heatmap") +
  theme(plot.title = element_text(family="Trebuchet MS", size=36, face="bold", hjust=0, color="#777777"))

berkeley_crime$date = as.Date(berkeley_crime$EVENTDT, format="%m/%d/%Y")

setwd("/Users/jason/Desktop/Quick-Data-Science-Experiments-2016/berkeley_crimes/timeseries_graphs")

i = 1
map.berkeley +
  stat_density2d(data=filter(berkeley_crime, OFFENSE == "DOMESTIC VIOLENCE", months(date) %in% month.name[i]), aes(x=latitude, y=longitude, color=..density.., size=ifelse(..density..<=1,0,..density..), alpha=..density..), geom="tile",contour=F) +
  scale_color_continuous(low="orange", high="red", guide = "none") +
  scale_size_continuous(range = c(0, 3), guide = "none") +
  scale_alpha(range = c(0,.5), guide="none") +
  ggtitle(paste("Berkeley Domestic Violence Heatmap - ", month.name[i])) +
  theme(plot.title = element_text(family="Trebuchet MS", size=36, face="bold", hjust=0, color="#777777"))


