# Applied Machine Learning for Health Data
# Assignment 2
# Author: Steve Hawley
# Date: Nov 17, 2019

####################################
########## DATA CLEANING ###########
####################################
library(summarytools)
library(epiR)

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
levels(cland$sex)[levels(cland$sex)=="0"] <- "F"
levels(cland$sex)[levels(cland$sex)=="1"] <- "M"
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
dfSummary(cland.cl)

####################################
###### LOGISTIC REGRESSION #########
####################################

library(glmnet)

set.seed(10)
acc.l <- c()

for (i in 1:5){

  train.I <- sample(nrow(cland.cl),nrow(cland.cl)*2/3)
  train <- cland.cl[train.I,]
  test <- cland.cl[-train.I,]
  
  x.train <- model.matrix(num ~.,train)[,-1]
  y.train <- train$num
  
  x.test <- model.matrix(num ~.,test)[,-1]
  y.test <- test$num
  
  l.mod <- glmnet(x.train,y.train,family="binomial",alpha=1) # LASSO regression
  cv.l <- cv.glmnet(x.train,y.train,alpha=1, family="binomial", type.measure = "mse") #optimal lambda
  #coef.min.l <- coef(cv.l, s = "lambda.min") # coefficients with optimal lambda
  
  predictions <- predict(l.mod, newx = x.test, type = "class", s= cv.l$lambda.min)

  t <- table(predictions, y.test)[2:1,2:1]
  epi.tests(t)
  acc.l[i] <- sum(diag(t))/sum(t) 
  }

# acc.l <- mean(acc.l)
# acc.l

#build model using k-folds cross validation (k=10 by default). Using LASSO regularization
# cv.l <- cv.glmnet(x,y,alpha=1,family = "binomial", type.measure = "mse")
# plot(cv.l,xvar="lambda",label = T)
# cv.l$lambda.min
# coef.min <- coef(cv.l, s = "lambda.min")
# coef.min
# 
# predictions.l <- predict(cv.l, newx = x, type = "class", s= cv.l$lambda.min)
# predictions.l.roc <- predict(cv.l, newx = x, type = "response", s= cv.l$lambda.min)
# myroc.l <- roc(y ~ predictions.l.roc, data=cland.cl)
# plot(myroc.l)

# t <- table(predictions.l, y)[2:1,2:1]
# epi.tests(t)
# acc <- sum(diag(t))/sum(t) 
# acc


####################################
############### KNN ################
####################################

library(class)

x <- model.matrix(num ~.,cland.cl)[,-1]
y <- cland.cl$num

x.sc <- scale(x)
 
# for (j in 1:50){ 
#   predictions <- knn.cv(x.sc,y,k=j) 
#   t <- table(predictions, y)[2:1,2:1] 
#   acc[j] <- sum(diag(t))/sum(t) 
#   }
# k.opt <- max(which(acc==max(acc))) 

set.seed(10)
acc.j <- c()
acc.k <- c()

for(i in 1:5){
train.I <- sample(nrow(cland.cl),nrow(cland.cl)*2/3)

train.data <- x.sc[train.I,]
test.data <- x.sc[-train.I,]
labels.train <- y[train.I]
labels.test <- y[-train.I]

for (j in 1:50){ 
  predictions <- knn.cv(train.data,y[train.I],k=j) 
  t <- table(predictions, y[train.I])[2:1,2:1] 
  acc.j[j] <- sum(diag(t))/sum(t) 
}

k.opt <- max(which(acc.j==max(acc.j))) 

predictions <- knn(train.data,test.data,labels.train,k=k.opt)
t <- table(predictions, labels.test)[2:1,2:1]
t
epi.tests(t)
acc.k[i] <- sum(diag(t))/sum(t) 
}

#acc.k <- mean(acc.k)
#acc.k

# KNN using LOO for cross-validation
# predictions.knn <- knn.cv(x.sc,y,k=10) 
# t <- table(predictions.knn, y)[2:1,2:1]
# epi.tests(t)
# acc <- sum(diag(t))/sum(t) 
# acc

####################################
####### CLASSIFICATION TREE ########
####################################

library(tree)
library(pROC)

# cfit2 <- tree(num ~., data = cland.cl) 
# summary(cfit2)
# 
# cv.res <- cv.tree(cfit2, FUN=prune.tree, method = "misclass", K = 5)
# cv.res
# 
# pruned <- prune.misclass(cfit2,best=5) 
# plot(pruned)
# text(pruned,pretty=0)
# 
# predictions.cvtree <- predict(pruned,newdata=cland.cl,type="class") 
# t <- table(predictions.cvtree, cland.cl$num)[2:1,2:1] #need to reformat the table for epi.tests
# epi.tests(t)
# acc <- sum(diag(t))/sum(t) #ex. (38 + 82) / 146
# acc



#Train the tree model
set.seed(10)
acc.t <- c()

for (i in 1:5){
  train.I <- sample(nrow(cland.cl),nrow(cland.cl)*2/3)
  tmp.tree <- tree(num ~. , data = cland.cl,subset=train.I)
  #plot(tmp.tree)
  #text(tmp.tree, pretty = 0)
  
  #prune the tree
  cv.train <- cv.tree(tmp.tree, FUN=prune.tree, method = "misclass", K = 5)
  pruned.train <- prune.misclass(tmp.tree,best=5) 
  #plot(pruned.train)
  #text(pruned.train,pretty=0)
  
  #test the model with test data
  preds3 <- predict(pruned.train,newdata=cland.cl[-train.I,],type="class")
  t <- table(preds3, y[-train.I])[2:1,2:1]
  epi.tests(t)
  acc.t[i] <- sum(diag(t))/sum(t) 
}

#acc.t <- mean(acc.t)

all.acc <- data.frame(LASSO=acc.l, KNN=acc.k, TREE=acc.t)
boxplot(all.acc, ylab="accuracy")
summary(all.acc)


library(reshape2)
m.acc <- melt(all.acc)
a1 <- aov(value~variable, data=m.acc)
summary(a1)
TukeyHSD(a1)
