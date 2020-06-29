

dane_proj3<-d.ibm3dxwkdays8008
View(dane_proj3)
ew_ts<-as.ts(dane_proj3$ew)
plot(ew_ts)
MONDAY<-as.factor(dane_proj3$M)
TUESDAY<-as.factor(dane_proj3$T)
WEDNESDAY<-as.factor(dane_proj3$W)
THURSDAY<-as.factor(dane_proj3$R)
FRIDAY<-as.factor(dane_proj3$F)
vw<-as.ts(dane_proj3$vw)
sp<-as.ts(dane_proj3$sp)



taks1_model1<- lm(ew_ts~MONDAY+TUESDAY+WEDNESDAY+THURSDAY+FRIDAY)
taks1_model1$coefficients


taks1_model2 <- lm(ew_ts~MONDAY+TUESDAY+WEDNESDAY+THURSDAY)
summary(taks1_model2)


plot(as.ts(taks1_model2$residual))

acf(taks1_model2$residuals)
pacf(taks1_model2$residuals)



arima_residuals<-auto.arima(taks1_model2$residuals)
summary(arima_residuals)


#######task 2
library(nnet)
data_task2 <- m.ge2608
View(data_task2)

data_task2<-as.data.frame(m.ge2608$rtn[4:996])
colnames(data_task2)="rtn"
data_task2$rt1 = data_task2$rtn[3:995]
data_task2$rt2 = data_task2$rtn[2:994]
data_task2$rt3 = data_task2$rtn[1:993]
training_data_task2<-data_task2[1:960,]
test_data_task2<-data_task2[961:993,]
model_nnet<-nnet(rtn~., data_task2[961:993,], size = 2)

mean((model_nnet$residuals)^2)
mean(( predict(model_nnet, data_task2[961:993,], type = "raw") - data_task2$rtn[961:993])^2)




data_task2$drtn1<-ifelse(data_task2$rtn1>0, 1 ,0)
data_task2$drtn2<-ifelse(data_task2$rtn2>0, 1 ,0)
data_task2$drtn3<-ifelse(data_task2$rtn3>0, 1 ,0)

model_nnet2<-nnet(rtn~., data_task2[961:993,], size = 2)
mean(( predict(model_nnet2, data_task2[961:993,], type = "raw") - data_task2$rtn[961:993])^2)


