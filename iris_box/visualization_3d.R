library(scatterplot3d)
names(mtcars)
scatterplot3d(mtcars$disp, mtcars$wt, mtcars$mpg, main="3-D Scatterplot Example 1")
scatterplot3d(mtcars$disp, mtcars$wt, mtcars$mpg, color="blue", pch=19, type="h", xlab="displacement", ylab="weight", zlab="miles / gallon")

pairs(~mpg+disp+drat+wt,data=mtcars, main="Simple Scatterplot Matrix")
