# Applied Machine Learning for Health Data
# Assignment 2
# Author: Steve Hawley
# Date: Nov 17, 2019

####################################
########## DATA CLEANING ###########
####################################

#load the data
cland <- read.csv(file = "cleveland.txt",header = F)

# examine the data
str(cland)
summary(cland)

#update headers
predictors <- c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal")
colnames(cland) <- c(predictors,"num")

#create factors
cland$sex <- factor(cland$sex) #sex
cland$cp <- factor(cland$cp) #chest pain type
cland$fbs <- factor(cland$fbs) #fasting blood sugar (T/F)
cland$restecg <- factor(cland$restecg) #resting ecg 0 = normal
cland$exang <- factor(cland$exang) #exercise induced angina (Y/N)
cland$slope <- factor(cland$slope) #slop of peak exercise ST segment

#clean up response to 0 = absence; 1-4 = presence of heart disease
cland$num <- factor(ifelse(cland$num == 0, "0", "1-4"), levels=c("0","1-4"))

#remove ? from the factors
cland$thal <- factor(cland$thal, levels = c("3.0","6.0","7.0"))
cland$ca <- factor(cland$ca, levels = c("0.0","1.0","2.0","3.0"))

#drop missing values
cland.cl <- na.omit(cland)

#review cleaned data
str(cland.cl)
summary(cland.cl)

####################################
###### LOGISTIC REGRESSION #########
####################################

library(glmnet)

x <- model.matrix(num ~.,cland.cl)[,-1]
y <- cland.cl$num

rr.mod <- glmnet(x,y,family="binomial",alpha=0) # ridge regression
plot(rr.mod,label = T)

cv.rr <- cv.glmnet(x,y,alpha=0,family = "binomial", type.measure = "mse")
plot(cv.rr) #plot is smoother than auc. Thus, Brier's is better to use here.
cv.rr$lambda.min
coef.min <- coef(cv.rr, s = "lambda.min")

#### DO MORE WORK HERE ####


####################################
############### KNN ################
####################################

library(class)
library(epiR)

x.sc <- scale(x)

set.seed(123)
train.I <- sample(nrow(x.sc),round(nrow(x.sc)*2/3))

train.data <- x.sc[train.I,]
test.data <- x.sc[-train.I,]
labels.train <- y[train.I]
labels.test <- y[-train.I]

predictions <- knn(train.data,test.data,labels.train,k=5)
t <- table(predictions, labels.test)[2:1,2:1] 
t
epi.tests(t)



####################################
####### CLASSIFICATION TREE ########
####################################

library(rpart)
library(tree)
library(epiR)
library(pROC)

cfit2 <- tree(num ~., data = cland.cl) #haven't classified regression or classification tree because outcome is categorical -- the function will figure it out
summary(cfit2)

plot(cfit2)
text(cfit2, pretty = 0)

# let's make predictions
preds0 <- predict(cfit2,type="class")

# what exactly does this vector contain?
# -- predictions for every sample

preds <- predict(cfit2,newdata=cland.cl,type="class") #this data set includes missing values
t <- table(preds, cland.cl$num)[2:1,2:1] #need to reformat the table for epi.tests
epi.tests(t)
acc <- sum(diag(t))/sum(t) #ex. (38 + 82) / 146
acc
