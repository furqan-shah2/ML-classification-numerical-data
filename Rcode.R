library(dplyr)

## Read EPL results and odds data retrieved from Kaggle.com
epl_data <- read.csv("final_dataset_with_odds.csv")


## keep selected variables and re-code FTR
epl_data <- epl_data %>%
  mutate(FTR = case_when(FTHG > FTAG ~ "H",
                         FTHG == FTAG ~ "D",
                         FTHG < FTAG ~ "A"),
         FTR = as.factor(FTR)) %>%
  select(FTR, B365H, B365D, B365A, IWH, IWD, IWA, LBH, LBD, LBA, WHH, WHD, WHA)

summary(epl_data)

## ML - Training
library(caret)
set.seed(1234)

train_row_numbers <- createDataPartition(epl_data$FTR, p=0.8, list=FALSE)

train_data <- epl_data[train_row_numbers,]
test_data <- epl_data[-train_row_numbers,]


set.seed(100) 
# Training the data using randomForest (rf)
library(randomForest)
model_rf = train(FTR ~ ., data=train_data, method='rf') 


# plotting variables importance 
varimp_rf2 <- varImp(model_rf) 
plot(varimp_rf2, main="Variable Importance with Random Forest 2")

# Predict on test_data 
predicted <- predict(model_rf, test_data)

# Confusion matrix to vaidate results 
confusionMatrix(reference = test_data$FTR, data = predicted, mode='everything')
