library(wikipediatrend)
library(AnomalyDetection)
library(ggplot2)
library(dplyr)

bigshort_film = wp_trend("The Big Short (film)", from="2005-01-01", lang = "en")
ggplot(bigshort_film, aes(x=date, y=count, color=count)) + geom_line() + ggtitle("Wiki page count for The Big Short (Movie 2015)")

bigshort_book = wp_trend("The Big Short", from="2005-01-01", lang = "en")
ggplot(bigshort_book, aes(x=date, y=count, color=count)) + geom_line() + ggtitle("Wiki page count for The Big Short (Book 2010)")

old_cdo = wp_trend("Collateralized debt obligation", from="2005-01-01", lang = "en")
ggplot(old_cdo, aes(x=date, y=count, color=count)) + geom_line() + ggtitle("Wiki page count for Collateralized debt obligation (caused 2008 Financial Crisis)")
old_cdo$formated_date = as.POSIXct(old_cdo$date)
old_cdo_anomaly = AnomalyDetectionTs(select(old_cdo, formated_date, count), max_anoms=0.01, direction="both", plot=TRUE, e_value = T, alpha=0.05)
old_cdo_anomaly$plot

mburry = wp_trend("Michael Burry", from="2005-01-01", lang = "en")
ggplot(mburry, aes(x=date, y=count, color=count)) + geom_line() + ggtitle("Wiki page count for Michael Burry (Investor who shorted against CDOs before the 2008 financial crisis)")
mburry$formated_date = as.POSIXct(mburry$date)
mburry_anomaly = AnomalyDetectionTs(select(mburry, formated_date, count), max_anoms=0.01, direction="both", plot=TRUE, e_value = T, alpha=0.05)
mburry_anomaly$plot

mbaum_actor = wp_trend("Steve Eisman", from="2005-01-01", lang = "en")
ggplot(mbaum_actor, aes(x=date, y=count, color=count)) + geom_line() + ggtitle("Wiki page count for the actor of Mark Baum (angry financier; my favorite character)")

bpitt = wp_trend("Brad Pitt", from="2005-01-01", lang = "en")
ggplot(bpitt, aes(x=date, y=count, color=count)) + geom_line() + ggtitle("Wiki page count for the actor of Brad Pitt (weird retired investor)")
