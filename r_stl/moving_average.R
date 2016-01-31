nottem

nottem.1 <- filter(nottem, filter=rep(1/1, 1))
nottem.2 <- filter(nottem, filter=rep(1/5, 5))
nottem.3 <- filter(nottem, filter=rep(1/3, 3))

plot(nottem.1)
plot(nottem.2)
plot(nottem.3)

