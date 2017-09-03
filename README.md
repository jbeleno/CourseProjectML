<style type="text/css">.main-container { max-width: 940px; margin-left: auto; margin-right: auto; } code { color: inherit; background-color: rgba(0, 0, 0, 0.04); } img { max-width:100%; height: auto; } .tabbed-pane { padding-top: 12px; } button.code-folding-btn:focus { outline: none; }</style>

<div class="container-fluid main-container"><script>$(document).ready(function () { window.buildTabsets("TOC"); });</script>

<div class="fluid-row" id="header">

# Course Project: Predicting ways of performing barebell fit with accelerometers data

#### _Juan Sebastián Beleño Díaz_

#### _3/9/2017_

</div>

<div id="executive-summary" class="section level2">

## Executive Summary

I built a Random Forest model to predict the way of performing barebell lift according to data collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The dataset is a subset of data collected from this [source](http://groupware.les.inf.puc-rio.br/har). My model seems to have a 100% of accuracy, which can be (or not) due to overfitting. However, I performed cross-validation with 5 folds to avoid that. Moreover, I did a simple data analysis and feature selection before using the data on the model.

</div>

<div id="getting-data" class="section level2">

## Getting Data

Training and testing dataset were provided in the description of the course project. I manually opened the dataset before coding and I was able to identify that many variables had white space and NA values. Hence, I decided to add the `na.string` parameter in the `read.csv` function to convert such string values in NA values recognized by R. Moreover, I print the dimensions of the dataset to have a baseline before performing feature selection: 19622 rows and 160 columns (variables).

```
training_data_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
raw_df <- read.csv(training_data_url, header=T, sep = ",", na.strings = c("", "NA"))
dim(raw_df)
```

```
## [1] 19622   160
```

</div>

<div id="feature-selection" class="section level2">

## Feature Selection

As I found many variables with NA values in the manual inspection, I decided to select variables that has less than 20% of NA values, because I assum that if we allow more than such threshold, we will get poor results. Moreover, I printed the dimension of the dataset after such feature selection, which gaves as result 60 variables. 100 variables were dropped after the feature selection. 62.5% of variables were removed from the dataset.

```
na_threshold = 0.2
accepted_size <- nrow(raw_df)*na_threshold
na_values_per_column <- colSums(is.na(raw_df))
tidy_df <- raw_df[, na_values_per_column < accepted_size] 
dim(tidy_df)
```

```
## [1] 19622    60
```

I also found that the values of the outcome variable (`classe`) are “almost” equally distributed. Thus, it is not necessary to balance the dataset.

```
library(plyr)
```

```
## Warning: package 'plyr' was built under R version 3.2.5
```

```
count(tidy_df$classe)
```

```
##   x freq
## 1 A 5580
## 2 B 3797
## 3 C 3422
## 4 D 3216
## 5 E 3607
```

</div>

<div id="splitting-the-datasets" class="section level1">

# Splitting the Datasets

I created a seed equal to `31337` in order to increase the reproducibility of my results. The initial dataset was separated in training and testing dataset. 75% of the data for training and 25% for testing. Moreover, a validation dataset was created using the variables (columns) present in the training dataset, but without the `classe` variable.

```
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.2.5
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.5
```

```
set.seed(31337)
inTrain = createDataPartition(tidy_df$classe, p = 3/4)[[1]]
training = tidy_df[ inTrain,]
testing = tidy_df[-inTrain,]

validation_data_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
columns <- colnames(training)
columns <- columns[columns != "classe"]
raw_validation <- read.csv(validation_data_url, header=T, sep = ",", na.strings = c("", "NA"))
validation <- raw_validation[, columns]
```

</div>

<div id="creating-the-predictor-model" class="section level1">

# Creating the Predictor Model

I decided to use Random Forest because it is one of the models that provides more precision. The overfitting was controlled by using cross-validation with 5 folds.

```
# Defining a general train control
tc <- trainControl(method = "cv", number = 5)

# Random Forest
model <- train(classe ~ .,data=training,method="rf",trControl= tc)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
model
```

```
## Random Forest 
## 
## 14718 samples
##    59 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11774, 11775, 11775, 11773, 11775 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9949044  0.9935548
##   41    1.0000000  1.0000000
##   81    0.9999321  0.9999141
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 41.
```

</div>

<div id="results" class="section level1">

# Results

After running the model on the testing datasets, the results present a 100% accuracy, which can be (or not) due to overfitting. However, I think the cross-validation works and the model actually has such accuracy.

```
# Prediction in the testing dataset
predictionRf <- predict(model, newdata = testing)

# Estimated accuracy out of the sample
confusionMatrix(predictionRf, testing$classe)$overall[1]
```

```
##  Accuracy 
## 0.9997961
```

Furthermore, the model was used to predict the outcome variable in the validation dataset.

```
# Predict `classe` in the validation dataset
predictionRfValidation <- predict(model, newdata = validation)
predictionRfValidation
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```

</div>

</div>

<script>// add bootstrap table styles to pandoc tables function bootstrapStylePandocTables() { $('tr.header').parent('thead').parent('table').addClass('table table-condensed'); } $(document).ready(function () { bootstrapStylePandocTables(); });</script> <script>(function () { var script = document.createElement("script"); script.type = "text/javascript"; script.src = "https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; document.getElementsByTagName("head")[0].appendChild(script); })();</script>