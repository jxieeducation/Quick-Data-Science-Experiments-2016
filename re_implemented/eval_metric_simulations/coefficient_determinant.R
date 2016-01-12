eruption.lm = lm(eruptions ~ waiting, data=faithful)

summary(eruption.lm)

# Multiple R-squared:  0.8115,	Adjusted R-squared:  0.8108 

# the values are similar because we are only using one variable