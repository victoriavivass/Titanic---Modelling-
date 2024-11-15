setwd("C:/Users/Victoria/OneDrive/Escritorio/Intro a datos/2nd assignment")

rm(list = ls())
library(ggplot2)
library("caTools")
library("rpart")
library("rpart.plot")
library("caret")
library("randomForest")
load("titanic_train.Rdata")
View(titanic.train)
set.seed(1)

##Preprocessing 

summary(titanic.train)

##Survived is already a boolean factor

#Changing Cabin to hasCabin
titanic.train$hasCabin = as.factor(sample(c(0,1), 668, replace = TRUE, prob = NULL))
titanic.train$hasCabin[titanic.train$Cabin == ""]<- 0
titanic.train$hasCabin[titanic.train$Cabin != ""]<- 1


summary(titanic.train)

#Remove things that don't matter/aren't usable:
#Ticket is useless, Cabin is instead going to be hasCabin, 
titanic.train[,"Ticket"]=NULL
titanic.train[,"Cabin"]=NULL

##Fixing labels to the boolean factor "Survived"

titanic.train$Survived <- factor(titanic.train$Survived, levels = c(0,1), labels = c("Died", "Survived"))

#Check
summary(titanic.train)

##First tree: without including hyperparameters

split <- sample.split(titanic.train$Survived, SplitRatio = 0.8)
trainingbest_set <- subset(titanic.train, split == TRUE)
testbest_set <- subset(titanic.train, split == FALSE)
firstTree <- rpart(formula=Survived ~., data=trainingbest_set, method="class")
prp(firstTree, 
    type= 1,
    extra=106, 
    box.palette="OrPu",
    shadow.col="gray", digits = 2, 
    nn=TRUE, roundint = FALSE)
pred <- predict(firstTree,testbest_set,type="class")
conf_matrix <- table(testbest_set$Survived,pred,dnn=c("Actual value","Classifier prediction"))
conf_matrix_prop <- prop.table(conf_matrix)

# Compute error estimates
#above 80% is good
accuracy = sum(diag(conf_matrix))/sum(conf_matrix)

precision = conf_matrix[1,1]/sum(conf_matrix[,1])

specificity = conf_matrix[2,2]/sum(conf_matrix[,2])

accuracy;precision;specificity

##Ploting variable importance 
firstTree$variable.importance

plottingdata <- data.frame(var_import = firstTree$variable.importance, var_names = names(firstTree$variable.importance))
g <- ggplot(plottingdata, aes(x = 1:length(plottingdata$var_import), y = var_import)) 
g + scale_x_continuous(breaks = 1:length(plottingdata$var_import), labels = plottingdata$var_names) + geom_line() + geom_point() + ggtitle("Variable Importance") + xlab("Decreasing variables order") + ylab("Relative Influence") + theme_minimal()


##Pruning the tree 

nfolds <- 10
folds <- createFolds(titanic.train$Survived, nfolds)

d_minsplit <- seq(2,40,by = 2)
d_maxdepth <- seq(1,5, by = 1)
d_cp <- 10^(-seq(2,4, by = 1))
                    
parametros <- expand.grid(misplit = d_minsplit, maxdepth = d_maxdepth, cp = d_cp);View(parametros)

cv_hyper <- apply(parametros, 1, function(y){
  cv <- lapply(folds, function(x){
    training_set <- titanic.train[-x,]
    test_set <- titanic.train[x,]
    tree <- rpart(Survived ~ ., training_set, method = "class", control = rpart.control(minsplit = y[1], 
                                                                                        maxdepth = y[2], cp = y[3]))
    pred <- predict(tree, test_set, type = "class")
    confusionmatrixx <- table(test_set$Survived, pred, dnn = c("Actual value", "Predicted value"))
    confusionmatrixxprop <- prop.table(confusionmatrixx)
    accuracy <- sum(diag(confusionmatrixx)/sum(confusionmatrixx))
    sensitivity <- confusionmatrixx[1,1]/sum(confusionmatrixx[,2])
    specificity <- confusionmatrixx[2,2]/sum(confusionmatrixx[,2])
    return(c(accuracy, sensitivity, specificity))
  })
  modqualres <- data.frame(t(matrix(unlist(cv), nrow = 3)))
  names(modqualres)=c("accuracy","sensitivity","specificity")
  return(c(mean(modqualres$accuracy),mean(modqualres$sensitivity),mean(modqualres$specificity)))
})

View(cv_hyper)
 
aux <- which.max(cv_hyper[1,])
parametros[aux,]#hyperparameters
cv_hyper[,aux] #maxaccuracy 

##Best model 

besttree <- rpart(Survived~., trainingbest_set, method = "class", control = rpart.control(maxdepth = 5, minsplit = 18, cp = 0.001))
prediction <- predict(besttree, testbest_set, type = "class")
confmatrix <- table(testbest_set$Survived, prediction, dnn = c("Actual value", "Predicted value"))
propconfmatrix <- prop.table(confmatrix)
baccuracy <- sum(diag(confmatrix))/sum(confmatrix)
bspecificity <- confmatrix[1,1]/sum(confmatrix[,1])
bsensitivity <- confmatrix[2,2]/sum(confmatrix[,2])
propconfmatrix                                    
baccuracy;bspecificity;bsensitivity 

prp(besttree, 
    type= 1,
    extra=106, 
    box.palette="OrPu",
    shadow.col="gray", digits = 2, 
    nn=TRUE, roundint = FALSE)

besttree$variable.importance

plottingbdata <- data.frame(var_import = besttree$variable.importance, var_names = names(besttree$variable.importance))
g <- ggplot(plottingbdata, aes(x = 1:length(plottingbdata$var_import), y = var_import)) 
g + scale_x_continuous(breaks = 1:length(plottingbdata$var_import), labels = plottingbdata$var_names) + geom_line() + geom_point() + ggtitle("Variable Importance") + xlab("Decreasing variables order") + ylab("Relative Influence") + theme_minimal()

##Repeated Validation for best model 

nrep <- 100
splits <- replicate(nrep, sample.split(titanic.train, SplitRatio = 0.8), simplify = FALSE)
?replicate
repeatedvalidation <- lapply(splits, function(x){
  trainingset <- subset(titanic.train, x == TRUE)
  testingset <- subset(titanic.train, x == FALSE)
  besttree <- rpart(Survived~., trainingset, method = "class", control = rpart.control(maxdepth = 5, minsplit = 18, cp = 0.001))
  prediction <- predict(besttree, testingset, type = "class")
  confmatrix <- table(testingset$Survived, prediction, dnn = c("Actual value", "Predicted value"))
  propconfmatrix <- prop.table(confmatrix)
  baccuracy <- sum(diag(confmatrix))/sum(confmatrix)
  bspecificity <- confmatrix[1,1]/sum(confmatrix[,1])
  bsensitivity <- confmatrix[2,2]/sum(confmatrix[,2])
  return(c(baccuracy, bsensitivity, bspecificity))
})

plotresults <- data.frame(values=unlist(repeatedvalidation),  parameter = as.factor(c(rep(c("accuracy",
                                                                                            "sensitivity","specificity"),nrep))))  

View(plotresults)

#Maximum Accuracy 

validation <- subset(plotresults, parameter == "accuracy", select = TRUE)
View(validation)
maxaccuracy <- which.max(validation$values)
validation$values[maxaccuracy] #0.8590604 is the max accuracy 

i <- ggplot(plotresults) + aes(x = parameter, y = values, fill = parameter) + geom_boxplot(outlier.colour = "Orange", outlier.shape = 17, outlier.size = 2.25) + scale_fill_brewer(palette = "Oranges") + theme_minimal() + theme(legend.position = "none")
i + ggtitle("Repeated Hold Out Validation Parameters", subtitle = "Decision Tree Technique") 

##Kfold-validation best model

kfold <- lapply(folds, function(x){
  training <- titanic.train[-x,]
  testing <- titanic.train[x,]
  besttree <- rpart(Survived ~., training, method = "class", control = rpart.control(minsplit = 12, maxdepth = 5, cp = 0.001))
  prediction <- predict(besttree, testing, type = "class")
  confmatrix <- table(testing$Survived, prediction, dnn = c("Actual value", "Predicted value"))
  propconfmatrix <- prop.table(confmatrix)
  baccuracy <- sum(diag(confmatrix))/sum(confmatrix)
  bspecificity <- confmatrix[1,1]/sum(confmatrix[,1])
  bsensitivity <- confmatrix[2,2]/sum(confmatrix[,2])
  return(c(baccuracy, bsensitivity, bspecificity))
})

plotresultskfold <- data.frame(values=unlist(kfold),
                               parameter=as.factor(c(rep(c("accuracy",
                                                           "sensitivity",
                                                           "specificity"),nfolds))))
View(plotresultskfold)

##Subsetting the accuracy 

kfoldaccuracy <- subset(plotresultskfold, parameter == "accuracy", select = TRUE)
View(kfoldaccuracy)
max <- which.max(kfoldaccuracy$values)
kfoldaccuracy$values [max] #0.9104478 is the max accuracy 

gg <- ggplot(plotresultskfold) + aes(x = parameter, y = values, fill = parameter) + geom_boxplot(outlier.colour = "Orange", outlier.shape = 17, outlier.size = 2.25) + scale_fill_brewer(palette = "Purples") + theme_minimal() + theme(legend.position = "none")
gg + ggtitle("K-fold Cross Validation Parameters", subtitle = "Decision Tree Technique") 

