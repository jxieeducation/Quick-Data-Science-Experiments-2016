library(ggplot2)
require(graphics)
require(acf)

kings <- scan("http://robjhyndman.com/tsdldata/misc/kings.dat",skip=3)
kingstimeseries <- ts(kings)

kingtimeseriesdiff1 <- diff(kingstimeseries, differences=1)
plot(kingstimeseries)
plot(kingtimeseriesdiff1)

acf(kingstimeseries, lag.max=20)
pacf(kingstimeseries, lag.max=20)


