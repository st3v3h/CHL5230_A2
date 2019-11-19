# Applied Machine Learning for Health Data
# Assignment 2
# Author: Steve Hawley
# Date: Nov 17, 2019

####################################
########## DATA CLEANING ###########
####################################

library(summarytools) #for descriptive statistics
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

#empty vector to collect accuracies
acc.l <- c()

#repeat validation 5 times
for (i in 1:5){
  
  #split into training and test sets
  train.I <- sample(nrow(cland.cl),nrow(cland.cl)*2/3)
  train <- cland.cl[train.I,]
  test <- cland.cl[-train.I,]
  
  x.train <- model.matrix(num ~.,train)[,-1]
  y.train <- train$num
  
  x.test <- model.matrix(num ~.,test)[,-1]
  y.test <- test$num
  
  #create the model
  l.mod <- glmnet(x.train,y.train,family="binomial",alpha=1) # LASSO regression
  #use cross validation to fine the optimal lambda
  cv.l <- cv.glmnet(x.train,y.train,alpha=1, family="binomial", type.measure = "mse") 
  coef.min.l <- coef(cv.l, s = "lambda.min")
  coef.min.l

  #test the model using optimal lambda
  predictions <- predict(l.mod, newx = x.test, type = "class", s= cv.l$lambda.min)

  #measure accuracy
  t <- table(predictions, y.test)[2:1,2:1]
  epi.tests(t)
  acc.l[i] <- sum(diag(t))/sum(t) 
  }

####################################
############### KNN ################
####################################

library(class)

#create model matrix and scale the predictors
x <- model.matrix(num ~.,cland.cl)[,-1]
y <- cland.cl$num
x.sc <- scale(x)
 
set.seed(10)

#create empty vector for KNN accuracies
acc.k <- c()

#repeat validation 5 times
for(i in 1:5){

  #set up training and test sets
  train.I <- sample(nrow(cland.cl),nrow(cland.cl)*2/3)
  train.data <- x.sc[train.I,]
  test.data <- x.sc[-train.I,]
  labels.train <- y[train.I]
  labels.test <- y[-train.I]

  #create empty vector to hold k values from cross validation
  acc.j <- c()

  #use cross validation to determine optimal k
  for (j in 1:50){ 
    predictions <- knn.cv(train.data,y[train.I],k=j) 
    t <- table(predictions, y[train.I])[2:1,2:1] 
    acc.j[j] <- sum(diag(t))/sum(t) 
  }

  #select k with highest accuracy 
  k.opt <- max(which(acc.j==max(acc.j))) 
  #plot(acc.j,type="l",xlab = "k") 

  #create model with optimal k
  predictions <- knn(train.data,test.data,labels.train,k=k.opt)
  t <- table(predictions, labels.test)[2:1,2:1]
  epi.tests(t)
  acc.k[i] <- sum(diag(t))/sum(t) 
}

####################################
####### CLASSIFICATION TREE ########
####################################

library(tree)

set.seed(10)

#create empty vector to hold tree accuracies
acc.t <- c()
bs <- c()
#repeat validation 5 times
for (i in 1:5){
  #train the tree model with training data
  train.I <- sample(nrow(cland.cl),nrow(cland.cl)*2/3)
  tmp.tree <- tree(num ~. , data = cland.cl,subset=train.I)
   plot(tmp.tree)
   text(tmp.tree, pretty = 0)
  
  #use cross validation to determine best size and prune the tree
  cv.train <- cv.tree(tmp.tree, FUN=prune.tree, method = "misclass", K = 5)
  plot(cv.train)
  best.size <- cv.train$size[which(cv.train$dev==min(cv.train$dev))]
  #collect tree sizes
  bs[i] <- best.size 
  pruned.train <- prune.misclass(tmp.tree,best=best.size) 
  plot(pruned.train)
  text(pruned.train,pretty=0)
  
  #test the pruned model with test data
  preds3 <- predict(pruned.train,newdata=cland.cl[-train.I,],type="class")
  t <- table(preds3, y[-train.I])[2:1,2:1]
  epi.tests(t)
  acc.t[i] <- sum(diag(t))/sum(t) 
}

####################################
####### MODEL COMPARISON ###########
####################################

#gather accuracies from all models into dataframe and compare
all.acc <- data.frame(LASSO=acc.l, KNN=acc.k, TREE=acc.t)
boxplot(all.acc, ylab="accuracy", xlab="method")
summary(all.acc)

#perform friedman test to check for significant differences
library(reshape2)
id <- rownames(all.acc)
all.acc <- cbind(id=id, all.acc)
m.acc <- melt(all.acc)
friedman.test(value~ variable | id, data = m.acc)
# a1 <- aov(value~variable + Error(id/variable), data=m.acc)
# summary(a1) 
# TukeyHSD(a1)

####################################
######## MODEL SELECTION ###########
####################################

#train model with full data set
l.mod.fin <- glmnet(x,y,family="binomial",alpha=1) 
#apply regularization
cv.l.fin <- cv.glmnet(x,y,alpha=1, family="binomial", type.measure = "mse") 
#get coefficients
coef.min.l <- coef(cv.l.fin, s = "lambda.min")
coef.min.l

#example prediction
#use a previously exluded observation for example
ndf <-cland[288,]
ndf[is.na(ndf)] <- "0.0"
ndf.x <- as.matrix(model.matrix(num ~.,ndf)[,-1])
ndf.x <- t(ndf.x)

#prediction
predict(l.mod.fin, newx = ndf.x, type = "class", s= cv.l$lambda.min)
#actual
ndf$num[1]



