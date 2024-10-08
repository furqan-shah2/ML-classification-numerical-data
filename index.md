Predicting EPL Full-Time Result using Random Forest
================
Your Name

## Introduction

In this tutorial, I will demonstrate how to use betting odds data from
various companies to predict the Full-Time Result (FTR) of English
Premier League (EPL) matches. The possible outcomes are: - **H**: Home
win - **D**: Draw - **A**: Away win

This is a simplified example where the primary objective is to showcase
how to implement a classification model using numerical data. Accuracy
is not the main focus here. We do not take into account team effects,
time effects, year effects, or any other additional contextual factors
that could improve the model’s predictive performance.

**The purpose** of this tutorial is to help you understand how the code
works so you can adopt it and tailor it to fit your specific needs. You
can apply this method to a wide range of research problems, such as
classifying narratives into different themes or tones in accounting
reports.

## Step 1: Load and Install Required Libraries

``` r
# Install necessary libraries if they aren't already installed
if (!require(dplyr)) install.packages("dplyr")
if (!require(caret)) install.packages("caret")
if (!require(randomForest)) install.packages("randomForest")
if (!require(e1071)) install.packages("e1071")  # e1071 is required by caret

# Load the libraries (make sure you have installed these packages before loading libraries)
library(dplyr)
library(caret)
library(randomForest)
```

In this first step, we ensure that the necessary libraries are
installed. I recommend running this code as it will install any missing
packages before proceeding with the tutorial. We select only following
variables for simplicity:

**FTHG** = Full Time Home Team Goals  
**FTAG** = Full Time Away Team Goals  
**FTR** = Full Time Result (H=Home Win, D=Draw, A=Away Win) OR (H=Home
Win, NH= Not a Home Win). Therefore, I create a new FTR variable based
on FTHG and FTAG variables  
**B365H**\* = Bet365 home win odds  
**B365D** = Bet365 draw odds  
**B365A** = Bet365 away win odds  
**IWH** = Interwetten home win odds  
**IWD** = Interwetten draw odds  
**IWA** = Interwetten away win odds  
**LBH** = Ladbrokes home win odds  
**LBD** = Ladbrokes draw odds  
**LBA** = Ladbrokes away win odds  
**WHH** = William Hill home win odds  
**WHD** = William Hill draw odds  
**WHA** = William Hill away win odds

Therefore, essentially we are using four companies betting odds to
predict outcome (FTR).

## Step 2: Prepare the Data

Here, we load the dataset and prepare it for analysis. I re-coded the
Full-Time Result (FTR) to represent H (Home win), D (Draw), or A (Away
win), using FTHG and GTAG variables. We also select the variables
representing betting odds from various companies.

``` r
# Load EPL results and betting odds data from a CSV file (I downloaded this data from kaggle.com)
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
common approach in machine learning to avoid overfitting.

``` r
# Set seed for reproducibility
set.seed(1234)

# Split the data into 80% training and 20% testing
train_row_numbers <- createDataPartition(epl_data$FTR, p = 0.8, list = FALSE)
train_data <- epl_data[train_row_numbers, ]
test_data <- epl_data[-train_row_numbers, ]
```

## Step 4: Train a Model using an algorithm (we use Random Forest)

In this step, we split the dataset into two parts. We use 80% of the
data to train our model and 20% to test its performance. This is a
common approach in machine learning to avoid overfitting.

``` r
# Set seed for reproducibility
set.seed(100)

# Train the model using Random Forest
model_rf <- train(FTR ~ ., data = train_data, method = 'rf')
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

![](Title_files/figure-gfm/Variable%20imp-1.png)<!-- -->

## Step 5: Make Predictions and Evaluate the Model

Now, we test the model using the test dataset and predict the Full-Time
Result (FTR). I use a confusion matrix to evaluate the model’s
performance, which tells us how well the model classifies each outcome.

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

## Step 5: Conclusion

The focus of this tutorial is to demonstrate how to implement a
classification model using numerical data. The model can be further
optimized by incorporating additional factors (such as team performance,
year effects, etc.), but for now, the goal is to show you the process so
you can apply it to your own context.

For example, in accounting research, this method can be adapted to
classify the tone of narratives in financial disclosures or categorize
different reporting themes. We are not aiming for high accuracy here but
rather to provide a basic example for learning and experimentation.
