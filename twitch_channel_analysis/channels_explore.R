library(ggplot2)
channels <- read.csv("/Users/jason/Downloads/twitch-info/channels.csv", header=TRUE)
names(channels)
head(channels, 5)

high_viewer <- channels[channels$viewers_max > 500,]
nrow(high_viewer) # this is 0.00458 of total (7049/1536350)
nrow(channels)

top_session_viewers <- sum(as.numeric(high_viewer$viewers_sum))
total_session_viewers <- sum(as.numeric(channels$viewers_sum))
# very top heavy in viewership (4468198667 / 5229740514)
viewership_df <- data.frame(name=c('views by 0.5% of streamers', 'views by all'), number_of_viewers=c(top_session_viewers, total_session_viewers))
ggplot(viewership_df, aes(x=name, y=number_of_viewers)) + geom_bar(stat="identity", width=0.8, fill="#FF9999", colour="black") + theme(text = element_text(size=15))

geo_df <- data.frame(location=c("total"), number_of_streamer=c(9999999))
geo_rank <- table(channels$geo)
for(loc in unique(channels$geo)){
  geo_df <- rbind(geo_df, c(as.character(loc), 1))
}
