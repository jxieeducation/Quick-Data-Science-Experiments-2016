nottem
log(nottem)
plot(stl(log(nottem), s.window="periodic"))

lnottem <- log(nottem)
t <- seq(1920, 1940, length=length(lnottem))
t2 <- t^2
plot(lnottem)
lm(lnottem~t+t2)
plot(t, lnottem, type="l", col="black")
lines(t, lm(lnottem~t+t2)$fit, col="red")

sin.t <- sin(2 * pi * t)
cos.t <- cos(2 * pi * t)
lines(t, lm(lnottem~t + t2 + sin.t + cos.t)$fit, col="blue")
