nottem
plot(nottem)
lnottem <- log(nottem)
plot(stl(log(nottem), s.window="periodic"))

# seasonal
t <- seq(1920, 1940, length=length(lnottem))
sin.t <- sin(2 * pi * t)
cos.t <- cos(2 * pi * t)
plot(lnottem)
lines(t, lm(lnottem~sin.t + cos.t)$fit, col="green")

# trend
t_2 <- t^2
t_3 <- t^3
plot(lnottem)
lines(t, lm(lnottem~t + t_2 + t_3)$fit, col="blue")

# residual
plot(lnottem)
lines(t, lm(lnottem~t + t_2 + t_3 + sin.t + cos.t)$fit, col="purple")
plot(t, lnottem - lm(lnottem~t + t_2 + t_3 + sin.t + cos.t)$fit + rep(mean(lnottem), length(t)), col="red", type="l")

