Machine Learning: Predicting EPL Results using Bettings Odds
================
Furqan Shah
2024-10-08

# Introduction

This tutorial demonstrates machine learning classification using
numerical data in the context of the English Premier League (EPL). We
predict the full-time result (Home win, Draw, Away win) based on betting
odds and compare their predictive power. The focus is on showing how to
implement a classification model, not on achieving high accuracy. Team
effects, time effects, and other contextual factors are not considered.

## Step 1: Load and Install Required Libraries

In this first step, we ensure that the necessary libraries are
installed. I recommend running this code as it will install any missing
packages before proceeding with the tutorial.

``` r
# Install necessary libraries if they aren't already installed
if (!require(dplyr)) install.packages("dplyr")
if (!require(caret)) install.packages("caret")
if (!require(randomForest)) install.packages("randomForest")
if (!require(e1071)) install.packages("e1071")  # e1071 is required by caret

# Load the libraries 
library(dplyr)
library(caret)
library(randomForest)
```

## Step 2: Prepare the Data

Now, we load the dataset and select the relevant variables (betting odds
and final game result) listed as follows:

- **FTHG**: *Full Time Home Team Goals*  
- **FTAG**: *Full Time Away Team Goals*  
- **FTR**: *Full Time Result (H=Home Win, NH=Not a Home Win)*  
- **B365H**: *Bet365 home win odds*  
- **B365D**: *Bet365 draw odds*  
- **B365A**: *Bet365 away win odds*  
- **IWH**: *Interwetten home win odds*  
- **IWD**: *Interwetten draw odds*  
- **IWA**: *Interwetten away win odds*  
- **LBH**: *Ladbrokes home win odds*  
- **LBD**: *Ladbrokes draw odds*  
- **LBA**: *Ladbrokes away win odds*  
- **WHH**: *William Hill home win odds*  
- **WHD**: *William Hill draw odds*  
- **WHA**: *William Hill away win odds*

Note that the **FTR** variable originally has two levels (*Home Win* or
*Not Home Win*). We recode it into three levels: *Home Win*, *Draw*, and
*Away Win*, to align with the odds data from the four companies: Bet365,
Ladbrokes, William Hill, and Interwetten, each offering odds for these
three outcomes.

``` r
# Load EPL results and betting odds data from a CSV file (available on the github repository of this page)
epl_data <- read.csv("final_dataset_with_odds.csv") %>%
  select(FTHG, FTAG, FTR, B365H, B365D, B365A, IWH, IWD, IWA, LBH, LBD, LBA, WHH, WHD, WHA)

# Recode the Full-Time Result (FTR) based on Full-Time Home Goals (FTHG) and Full-Time Away Goals (FTAG)
epl_data <- epl_data %>%
  mutate(FTR = case_when(
    FTHG > FTAG ~ "H",  # Home win
    FTHG == FTAG ~ "D", # Draw
    FTHG < FTAG ~ "A")  # Away Win
    ) %>%
  select(-FTHG, -FTAG)

# Convert FTR to a factor variable
epl_data$FTR <- as.factor(epl_data$FTR)

# check data for any inconsistencies or missing values
summary(epl_data)
```

    ##  FTR          B365H            B365D            B365A             IWH        
    ##  A:1699   Min.   : 1.080   Min.   : 2.500   Min.   : 1.160   Min.   : 1.050  
    ##  D:1514   1st Qu.: 1.670   1st Qu.: 3.250   1st Qu.: 2.500   1st Qu.: 1.650  
    ##  H:2787   Median : 2.150   Median : 3.500   Median : 3.500   Median : 2.100  
    ##           Mean   : 2.702   Mean   : 3.891   Mean   : 4.829   Mean   : 2.508  
    ##           3rd Qu.: 2.870   3rd Qu.: 4.000   3rd Qu.: 5.500   3rd Qu.: 2.650  
    ##           Max.   :23.000   Max.   :13.000   Max.   :34.000   Max.   :15.000  
    ##       IWD              IWA              LBH             LBD        
    ##  Min.   : 2.500   Min.   : 1.200   Min.   : 1.08   Min.   : 2.750  
    ##  1st Qu.: 3.200   1st Qu.: 2.500   1st Qu.: 1.66   1st Qu.: 3.250  
    ##  Median : 3.300   Median : 3.200   Median : 2.10   Median : 3.400  
    ##  Mean   : 3.645   Mean   : 4.185   Mean   : 2.59   Mean   : 3.748  
    ##  3rd Qu.: 3.800   3rd Qu.: 4.600   3rd Qu.: 2.75   3rd Qu.: 3.750  
    ##  Max.   :10.500   Max.   :29.000   Max.   :21.06   Max.   :12.590  
    ##       LBA              WHH             WHD             WHA        
    ##  Min.   : 1.170   Min.   : 1.06   Min.   : 2.80   Min.   : 1.140  
    ##  1st Qu.: 2.450   1st Qu.: 1.66   1st Qu.: 3.20   1st Qu.: 2.500  
    ##  Median : 3.300   Median : 2.15   Median : 3.30   Median : 3.300  
    ##  Mean   : 4.498   Mean   : 2.63   Mean   : 3.65   Mean   : 4.549  
    ##  3rd Qu.: 5.000   3rd Qu.: 2.75   3rd Qu.: 3.75   3rd Qu.: 5.000  
    ##  Max.   :32.700   Max.   :17.00   Max.   :13.00   Max.   :29.000

## Step 3: Split Data into Training and Testing Sets

In this step, we split the dataset into two parts. We use 80% of the
data to train our model and 20% to test its performance. This is a
common approach in machine learning to avoid over-fitting and test model
efficiency.

``` r
# Set seed for reproducibility
set.seed(1234)

# Split the data into 80% training and 20% testing
train_row_numbers <- createDataPartition(epl_data$FTR, p = 0.8, list = FALSE)
train_data <- epl_data[train_row_numbers, ]
test_data <- epl_data[-train_row_numbers, ]
```

## Step 4: Train a Model using an algorithm (we use Random Forest)

While many algorithms are available, we will use Random Forest for its
simplicity and ability to highlight variable importance. This will allow
us to compare which betting odds have the strongest explanatory power.

``` r
# Set seed for reproducibility
set.seed(100)

# Train the model using Random Forest
model_rf <- train(FTR ~ ., data = train_data, method = 'rf')

# display model
model_rf
```

    ## Random Forest 
    ## 
    ## 4802 samples
    ##   12 predictor
    ##    3 classes: 'A', 'D', 'H' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 4802, 4802, 4802, 4802, 4802, 4802, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.4940229  0.1798234
    ##    7    0.4832901  0.1685693
    ##   12    0.4803413  0.1656557
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

Which betting companies’ odds have the best explaining power?

``` r
# Display the variable importance (which betting companies' odds have the best explaining power?)
varimp_rf <- varImp(model_rf)
plot(varimp_rf, main = "Horserace of Betting Odds")
```

![](index_files/figure-gfm/Variable%20imp-1.png)<!-- -->

## Step 5: Make Predictions and Evaluate the Model

Now, we test the model on the test dataset and predict the Full-Time
Result (FTR). To evaluate the model’s performance, we use a confusion
matrix, which provides insight into how accurately the model classifies
each outcome.

``` r
# Make predictions on the test set
predicted <- predict(model_rf, test_data)

# Generate a confusion matrix to validate the results
confusion_matrix <- confusionMatrix(reference = test_data$FTR, data = predicted, mode = 'everything')
confusion_matrix
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   A   D   H
    ##          A 139  78  87
    ##          D  56  49  60
    ##          H 144 175 410
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4992          
    ##                  95% CI : (0.4705, 0.5279)
    ##     No Information Rate : 0.4649          
    ##     P-Value [Acc > NIR] : 0.009547        
    ##                                           
    ##                   Kappa : 0.1797          
    ##                                           
    ##  Mcnemar's Test P-Value : 6.073e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: D Class: H
    ## Sensitivity            0.4100   0.1623   0.7361
    ## Specificity            0.8079   0.8705   0.5023
    ## Pos Pred Value         0.4572   0.2970   0.5624
    ## Neg Pred Value         0.7763   0.7551   0.6866
    ## Precision              0.4572   0.2970   0.5624
    ## Recall                 0.4100   0.1623   0.7361
    ## F1                     0.4323   0.2099   0.6376
    ## Prevalence             0.2830   0.2521   0.4649
    ## Detection Rate         0.1160   0.0409   0.3422
    ## Detection Prevalence   0.2538   0.1377   0.6085
    ## Balanced Accuracy      0.6090   0.5164   0.6192

## Conclusion

This tutorial provide a simple context to implement machine learning
classification using numerical predictors. The focus of this tutorial is
not on accuracy but rather on providing a basic example with code for
learning and experimentation. The model can be further optimized by
incorporating additional factors (such as team performance, year
effects, etc.) and can be adopted to one’s own research
contexts/settings.
