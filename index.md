### Predict human activity recognition  from accelerometers
### Introduction
Human Activity Recognition has emerged as a key research area in the last years, especially for the development of context-aware systems. This project is to predict har by data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 


### Exploratory data analysis
The data is pml-training.csv provide by "Practical Machine Learning" course. The data includes 19622 observations, each observation includes 160 aspects.

```r
d = read.csv("pml-training.csv")
dim(d)
```

```
## [1] 19622   160
```

There are many NA or empty observations in dataset. The ratio of NA or empty compare with full data is 0.9793, that means this aspect is useless. 

```r
column_count = dim(d)[2]
empty_na_columns = c()
large_empty_na_columns = c()
empty_na_ratio = c()
for (i in c(1:column_count)) {
    empty_na_count = length(d[, i][d[, i] == ""])
    data_count = length(d[, i])
    if (empty_na_count > 0) {
        empty_na_columns = append(empty_na_columns, names(d)[i])
        empty_na_ratio = append(empty_na_ratio, empty_na_count/data_count)
    }
    if (empty_na_count/data_count > 0.5) {
        large_empty_na_columns = append(large_empty_na_columns, names(d)[i])
    }
}
summary(empty_na_ratio)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   0.979   0.979   0.979   0.979   0.979   0.979
```

I remove all aspects which has many NA or empty values. Use R nearZeroVar function to check left aspects show new_window is a near zero variable. The X is row number , the user_name, raw_timestamp_part_1, raw_timestamp_part_2 and cvtd_timestamp are user name and test time, so clearly these variables have no effect on har and may make the model work worse. I remove all these variables and create a new clean data. This clean data has 19622 observations with 54 variables.

```r
d_removeNA = d
for (i in empty_na_columns) {
    d_removeNA[, i] = NULL
}
nsv = nearZeroVar(d_removeNA)
cleandata = d_removeNA
cleandata = cleandata[, -nsv]
cleandata$X = NULL
cleandata$user_name = NULL
cleandata$raw_timestamp_part_1 = NULL
cleandata$raw_timestamp_part_2 = NULL
cleandata$cvtd_timestamp = NULL
dim(cleandata)
```

```
## [1] 19622    54
```

Because the dataset is too large for my computer, I split data to training and validating by 40 - 60 for cross validation. The training dataset has 7850 observations, the validating dataset has 11772 observations.

```r
set.seed(56723)
inTrain = createDataPartition(y = cleandata$classe, p = 0.4, list = FALSE)
training = cleandata[inTrain, ]
validating = cleandata[-inTrain, ]
dim(training)[1]
```

```
## [1] 7850
```

```r
dim(validating)[1]
```

```
## [1] 11772
```

## Results
There are 54 variables in the dataset, it's diffcult to choose goof model for regression method. I use tree method to classify the har, and compare difference method performance. The methods used in this project are: C5.0, random forest with bootstrap, and random forest with 10 fold cross vaildation.

```r
"C50 Result"
```

```
## [1] "C50 Result"
```

```r
set.seed(56723)
ptm = proc.time()
model_c50 = train(classe ~ ., data = training, method = "C5.0")
# model_c50
pred = predict(model_c50, validating)
result_c50 = confusionMatrix(validating$classe, pred)
result_c50
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    3 2275    0    0    0
##          C    0    0 2049    4    0
##          D    0    0    0 1926    3
##          E    0    1    0    3 2160
## 
## Overall Statistics
##                                         
##                Accuracy : 0.999         
##                  95% CI : (0.998, 0.999)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    1.000    1.000    0.996    0.999
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    0.999    0.998    0.998    0.998
## Neg Pred Value          1.000    1.000    1.000    0.999    1.000
## Prevalence              0.285    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    0.998    0.999
```

```r
proc.time() - ptm
```

```
##    user  system elapsed 
##  2171.4     0.8  2178.3
```

```r

"random forest with bootstrap"
```

```
## [1] "random forest with bootstrap"
```

```r
ptm = proc.time()
model_rf_default = train(classe ~ ., data = training, method = "rf", importance = TRUE)
# model_rf_default
pred = predict(model_rf_default, validating)
result_rf_default = confusionMatrix(validating$classe, pred)
result_rf_default
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3347    1    0    0    0
##          B    8 2268    2    0    0
##          C    0   10 2040    3    0
##          D    0    0   30 1898    1
##          E    0    0    1    3 2160
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.994, 0.996)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.994         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.995    0.984    0.997    1.000
## Specificity             1.000    0.999    0.999    0.997    1.000
## Pos Pred Value          1.000    0.996    0.994    0.984    0.998
## Neg Pred Value          0.999    0.999    0.997    0.999    1.000
## Prevalence              0.285    0.194    0.176    0.162    0.184
## Detection Rate          0.284    0.193    0.173    0.161    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.997    0.991    0.997    1.000
```

```r
proc.time() - ptm
```

```
##    user  system elapsed 
## 4163.40   15.77 4214.73
```

```r

"random forest with 10 fold cross vaildation"
```

```
## [1] "random forest with 10 fold cross vaildation"
```

```r
ptm = proc.time()
model_rf_cv_1 = train(classe ~ ., data = training, method = "rf", trControl = trainControl(method = "cv", 
    number = 10), importance = TRUE)
# model_rf_cv_1
pred = predict(model_rf_cv_1, validating)
result_rf_cv = confusionMatrix(validating$classe, pred)
result_rf_cv
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3347    1    0    0    0
##          B    8 2267    3    0    0
##          C    0   10 2043    0    0
##          D    0    0   27 1902    0
##          E    0    1    1    3 2159
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.994, 0.997)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.994         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.995    0.985    0.998    1.000
## Specificity             1.000    0.999    0.999    0.997    0.999
## Pos Pred Value          1.000    0.995    0.995    0.986    0.998
## Neg Pred Value          0.999    0.999    0.997    1.000    1.000
## Prevalence              0.285    0.194    0.176    0.162    0.183
## Detection Rate          0.284    0.193    0.174    0.162    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.997    0.992    0.998    1.000
```

```r
proc.time() - ptm
```

```
##    user  system elapsed 
## 1418.76    4.09 1430.41
```

All tree methods give great result for this data. The R confusionMatrix function shows cross validation reslut. Accuracy for C50 is 0.999, for random forest with bootstrap is 0.995, for random forest with 10 fold cross vaildation is 0.995, with high p-value. The expected out of sample error is : 0.1% for C50, 0.5% for random forest with bootstrap, 0.5% for random forest with 10 fold cross vaildation. Random forest with bootstrap is slower than C50 method, random forest with cross valdation is the fastest.
To verify the model, I test it on the test dataset. All models give the same result on test dataset, and show 100% accuracy.

```r
testdata = read.csv("pml-testing.csv")
pred_rf_default = predict(model_rf_default, testdata)
pred_c50 = predict(model_c50, testdata)
pred_rf_cv1 = predict(model_rf_cv_1, testdata)
setdiff(pred_rf_default, pred_c50)
```

```
## character(0)
```

```r
setdiff(pred_c50, pred_rf_cv1)
```

```
## character(0)
```

## References
1) Eduardo Velloso. "Qualitative Activity Recognition of Weight Lifting Exercises". [http://perceptual.mpi-inf.mpg.de/files/2013/03/velloso13_ah.pdf]

