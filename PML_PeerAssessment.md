---
title: "PML_PeerAssessment"
author: "Owen"
date: "1/8/2021"
output: html_document
---

Practical Machine Learning - Assignment
======================================================================================

# 1. Packages

```{r These are R packages I will need, results="hide"}
library(fscaret)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(rattle)
library(randomForest)
library(RColorBrewer)
library(gbm)
```

# 2. Reading Data

```{r Reading Data}
data_train <- read.csv("C:/Users/Owenm/Documents/PML_PeerAssessment/pml-training.csv", strip.white = T, na.strings = c("NA",""))
data_ogtest <- read.csv("C:/Users/Owenm/Documents/PML_PeerAssessment/pml-testing.csv", strip.white = T, na.strings = c("NA",""))

dim(data_train)
dim(data_ogtest)
```

# 3. Data splitting and Feature Selecting
```
I set the seed to make this analysis reproducible
```
```{r Splitting Data and Selecting Features}
set.seed(200)
```
```
I split the original set into training and test sets (0.7 and 0.3 respectively)
```
```{r}
in_Train <- createDataPartition(y=data_train$classe, p=0.7, list = F)
train_set <- data_train[in_Train, ]
test_set <- data_train[-in_Train, ]

dim(train_set)
dim(test_set)
```
```
The two datasets (train_set and test_set) have a large number of NA, NZV (near-zero variance predicitors) variables that bring almost no information model and will make computing unnecessarily longer. Both variables are removed together with their ID variables.
```
```{r Removing NA and NZV}
#removing near-zero variance predictors
nzv <- nearZeroVar(data_train)
train_set <- train_set[, -nzv]
test_set <- test_set[, -nzv]

#removing predictors with NA values
train_set <- train_set[, colSums(is.na(train_set)) == 0]
test_set <- test_set[, colSums(is.na(test_set)) == 0]

#removing columns unfit for prediction (ID, user_name, raw_timestamp_part_1 etc ...)
train_set <- train_set[, -(1:5)]
test_set <- test_set[, -(1:5)]

dim(train_set)
dim(test_set)
```

# 4. Correlation Analysis
```
Correlation analysis between the variables before the modeling work itself is done. The “FPC” is used as the 
first principal component order.
```
```{r Correlation Analysis}
mcorr_matrix <- cor(train_set[ , -54])
corrplot(mcorr_matrix, order = "FPC", method = "circle", type = "lower",
         tl.cex = 0.6, tl.col = rgb(0, 0, 0))
```
```
If two variables are highly correlated their colors are either dark blue (for a positive correlation) or dark red (for a negative correlations). Because there are only few strong correlations among the input variables, the Principal Components Analysis (PCA) will not be performed in this analysis. Instead, a few different prediction models will be built to have a better accuracy.
```

# 5. Predicition Models

## Decision Tree Model

```{r Decision Tree Model}
set.seed(2000)
fit_decision_tree <- rpart(classe ~ ., data = train_set, method="class")
fancyRpartPlot(fit_decision_tree)
```
```
Predictions of the decision tree model on test_set
```
```{r Predictions of the decision tree model on test_set}
predict_decision_tree <- predict(fit_decision_tree, newdata = test_set, type="class")
conf_matrix_decision_tree <- confusionMatrix(predict_decision_tree, factor(test_set$classe))
conf_matrix_decision_tree
```
```
The predictive accuracy of the decision tree model is relatively low at 72.34 %.
```
```
Plot the predictive accuracy of the decision tree model
```
```{r Plot the predictive accuracy of the decision tree model}
plot(conf_matrix_decision_tree$table, col = conf_matrix_decision_tree$byClass, 
     main = paste("Decision Tree Model: Predictive Accuracy =",
                  round(conf_matrix_decision_tree$overall['Accuracy'], 4)))
```

## General Boosted Model (GBM)
```{r General Boosted Model (GBM)}
set.seed(2000)
ctrl_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_GBM  <- train(classe ~ ., data = train_set, method = "gbm",
                  trControl = ctrl_GBM, verbose = FALSE)
fit_GBM$finalModel
```

```
Predictions of the GBM on test_set
```
```{r Predictions of the GBM on test_set}
predict_GBM <- predict(fit_GBM, newdata = test_set)
conf_matrix_GBM <- confusionMatrix(predict_GBM, factor(test_set$classe))
conf_matrix_GBM
```

```
The predictive accuracy of the GBM is relatively high at 98.9 %.
```

## Random Forest Model
```{r Random Forest Model}
set.seed(2000)
ctrl_RF <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_RF  <- train(classe ~ ., data = train_set, method = "rf",
                  trControl = ctrl_RF, verbose = FALSE)
fit_RF$finalModel
```

```
Predictions of the random forest model on test_set
```
```{r Predictions of the random forest model on test_set}
predict_RF <- predict(fit_RF, newdata = test_set)
conf_matrix_RF <- confusionMatrix(predict_RF, factor(test_set$classe))
conf_matrix_RF
```
```
The predictive accuracy of the Random Forest model is excellent at 99.86 %.
```

# 6. Testing Model
```
The following are the predictive accuracy of the three models:

        * Decision Tree Model: 72.34 %
        * Generalized Boosted Model: 98.9 %
        * Random Forest Model: 99.86 %
        
The Random Forest model is selected and applied to make predictions on the 20 data points from the original testing dataset (test_set).
```

```{r Predictions on 20 Data Points with RFM}
final_rfm <- as.data.frame(predict(fit_RF, newdata = data_ogtest))
final_rfm
```
```
