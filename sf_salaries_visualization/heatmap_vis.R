salaries <- read.csv("/Users/jason/Downloads/sf_salaries/Salaries.csv")
small_salaries <- salaries[sample(nrow(salaries), 50, replace=FALSE),]

head(small_salaries, 10)
library(dplyr)
labels <- select(small_salaries, JobTitle)
formated_salaries <- select(small_salaries, BasePay, OvertimePay, OtherPay, Benefits, TotalPay, TotalPayBenefits)

formated_salaries$BasePay <- as.numeric(formated_salaries$BasePay)
formated_salaries$OvertimePay <- as.numeric(formated_salaries$OvertimePay)
formated_salaries$OtherPay <- as.numeric(formated_salaries$OtherPay)
formated_salaries$Benefits <- as.numeric(formated_salaries$Benefits)
formated_salaries$TotalPay <- as.numeric(formated_salaries$TotalPay)
formated_salaries$TotalPayBenefits <- as.numeric(formated_salaries$TotalPayBenefits)

salary_matrix <- as.matrix(scale(formated_salaries))
labels$str <- as.character(labels$JobTitle)
heatmap(salary_matrix, Colv=F, scale='none', margins=c(12, 2), labRow=labels$str)
