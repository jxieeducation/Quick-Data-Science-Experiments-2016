
# we do exponential smoothing to fit the future

nottem
t <- seq(1920, 1940, length=length(lnottem))

HoltWinters(nottem)
plot(nottem, type="l")
# [1] "xhat"   "level"  "trend"  "season"
lines(HoltWinters(nottem)$fitted[,1], col='red')
lines(HoltWinters(nottem)$fitted[,2], col='blue')
