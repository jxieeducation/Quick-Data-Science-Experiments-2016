sessions <- read.csv("/Users/jason/Downloads/twitch-info/sessions.csv", header=TRUE)

popular_sessions <- sessions[sessions$viewers_max > 500,]
unpopular_sessions <- sessions[sessions$viewers_max < 500, ]
nrow(popular_sessions)
nrow(unpopular_sessions)

popular_sessions <- popular_sessions[as.character(popular_sessions$geo) != "",]
popular_sessions <- popular_sessions[as.character(popular_sessions$geo) != "None",]

library("ggmap")
library(maptools)
library(maps)
mapping <- geocode(unique(as.character(popular_sessions$timezone)))
mapping$location <- unique(as.character(popular_sessions$timezone))
mapping$timezone <- mapping$location
head(mapping, 5)

library(plyr)
mapped_session <- join(popular_sessions, mapping, type="inner")
mapped_session <- mapped_session[!is.na(mapped_session$lon),]
head(mapped_session, 100)
names(mapped_session)

world_data <- map_data("world")
mp <- NULL
mapWorld <- borders("world", colour="gray50", fill="gray50") # create a layer of borders
mp <- ggplot() +   mapWorld
mp <- mp+ geom_point(aes(x=mapped_session$lon, y=mapped_session$lat) ,color="blue", size=3) + ggtitle("Popular Streamers")
#mp

mapped_unpopular_session <- join(unpopular_sessions, mapping, type="inner")
mapped_unpopular_session <- mapped_unpopular_session[!is.na(mapped_unpopular_session$lon),]
head(mapped_unpopular_session, 100)
nrow(mapped_unpopular_session)
small_unpopular_session <- sample(mapped_unpopular_session, size=5000, replace=TRUE)
mp <- NULL
mapWorld <- borders("world", colour="gray50", fill="gray50") # create a layer of borders
mp <- ggplot() +   mapWorld
mp <- mp+ geom_point(aes(x=small_unpopular_session$lon, y=small_unpopular_session$lat) ,color="red", size=3) + ggtitle("Unpopular Streamers")
mp
