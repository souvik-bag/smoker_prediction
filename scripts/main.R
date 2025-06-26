library(dplyr)
library(tidyverse)
library(ggplot2)
library(tidyr)
library(magrittr)
library("glmnet")
library(caret)
library(pROC)
library(Metrics)
library(MASS)
library(xgboost)
library(randomForest)
library(caTools)
library(corrplot)
library(partykit)

# import the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")
str(train)

# convert smoking status to a factor 

columns_to_convert <- c("smoking", "hearing.left.", "hearing.right.", "Urine.protein","dental.caries")  # Replace with your column names

train <- train %>% mutate_at(vars(columns_to_convert), as.factor)
test <- test %>%  mutate_at(vars(setdiff(columns_to_convert,"smoking")), as.factor)
# Discard the ID variable

train <-  subset(train, select = - c(id))
test <- subset(test, select = - c(id))
## Train Test split
training.sample <- train$smoking %>% createDataPartition(p = 0.99 , list = FALSE) 

train.data <- train[training.sample,]
test.data <- train[-training.sample,]




# Plot the data wrt smoking status

response_variable <- "smoking" # Replace with the actual response variable name
columns_to_plot <- setdiff(names(train), c(response_variable, "id",columns_to_convert)) # Select columns to plot

plots <- list() # Create an empty list to store plots

for (col in columns_to_plot) {
  p <- ggplot(train, aes_string(x = col , col = response_variable)) +
    geom_boxplot()+
    labs(title = paste( col, "vs", response_variable)) +
    theme(axis.text = element_text(size = 10),  # Adjusts the size of axis labels
          axis.title = element_text(size = 8))  # Adjusts the size of axis titles
  plots[[col]] <- p
}

# Display all the plots
multiplot <- do.call(gridExtra::grid.arrange, plots)
multiplot
 
ggplot(train, aes_string(x = "height.cm.", y = "weight.kg.")) +
  geom_point()
cor(train$height.cm.,train$weight.kg.)
cormat = cor(train[,columns_to_plot])

corrplot(cormat, method = "color")


# Fit a logistic regression

logit_model1 <- glm(smoking~., data = train.data, family = "binomial")
summary(logit_model1)

predicts <- predict(logit_model1, newdata = test.data, type = "response")

Metrics :: auc(test.data$smoking, predicts)



############# LASSO & RIDGE ###########


Y.train = train.data$smoking
X.train = model.matrix(smoking~., data = train.data)[,-1]
X.test = model.matrix(smoking~., data = test.data)[,-1]
Y.test = test.data$smoking

XY.train <- cbind(Y.train,X.train)

# First we get the sequence of lambdas


cv.lasso <- cv.glmnet(X.train, Y.train, alpha = 0, family = "binomial")

cv.lasso$lambda.min

# Fit the  model on the data
lasso_model <- glmnet(X.train, Y.train, alpha = 0, family = "binomial",
                      lambda = cv.lasso$lambda.min, intercept = F)

summary(lasso_model)

predicts <- predict(lasso_model, newx = X.test, type = "response")

Metrics :: auc(Y.test, predicts)

coef(lasso_model)




## random forest


rf_model <- randomForest(smoking ~., data = train.data, ntrees = 100, importance = TRUE)
predicts_rf <- predict(rf_model, newdata = test.data, type = "prob")
Metrics :: auc(Y.test, predicts_rf[,2])


# Extract variable importance for plotting
importance_df <- data.frame(
  Variable = row.names(importance(rf_model)),
  Importance = importance(rf_model)[,4]
)

# Sort the dataframe by Importance in descending order
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]

# Create a colorful variable importance plot using ggplot2
ggplot(importance_df, aes(y = reorder(Variable, Importance), x = Importance, fill = Importance)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Variable Importance Plot", x = "Variable", y = "Importance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# AUC for Random Forest 0.8532496

# traina nd test matrix

trainm <- sparse.model.matrix(smoking~., data = train.data)
train_label <- train.data[,"smoking"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = as.integer(train_label)-1 )

testm <- sparse.model.matrix(smoking~., data = test.data)
test_label <- test.data[,"smoking"]
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = as.integer(test_label)-1 )


# parameters


nc <- length(unique(train_label))

xgb_params <- list( "objective" = "multi:softprob", "eval_metric" = "mlogloss",
                    "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)

bst_model2 <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 500,
                       watchlist = watchlist,
                       eta = 0.09,
                       max.depth = 7,
                       gamma = 2,
                       subsample = 1,
                       )
 
p<- predict(bst_model, newdata = test_matrix)
head(p)

pred <- matrix(p, ncol = 2, byrow = T) %>% data.frame() 
Metrics :: auc(test_label, pred$X2)

# Train and test error plot 

e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = "blue", xlab = "Iteration", ylab = "Training logloss", main = " Training and testing log loss vs iteration")
lines(e$iter, e$test_mlogloss, col = "red")
min(e$test_mlogloss)
e[e$test_mlogloss == 0.4234187,]

# Submission prediction

test_matrix_sub <- sparse.model.matrix(~.,data =test)
test_matrix_sub <- xgb.DMatrix(data = as.matrix(test_matrix_sub))
p2<- predict(bst_model, newdata = test_matrix_sub)
pred2 <- matrix(p2, ncol = 2, byrow = T) %>% data.frame() 

results_submision <- cbind("id" = test$id, "smoking" =  pred2$X2)

write.csv(results_submision, file = "results_sub.csv",row.names = FALSE)

## Fit a neural network
Sys.setenv(RETICULATE_PYTHON = "/Users/souvikbag/Library/CloudStorage/OneDrive-UniversityofMissouri/Assignments/Fall 23/STAT 8330/Project1_python/.venv/bin/python")

# import library

library(keras)
library(tensorflow)

# read the data

train <- read.csv("/Users/souvikbag/Library/CloudStorage/OneDrive-UniversityofMissouri/Assignments/Fall 23/STAT 8330/Project 1/Project 1/train.csv")
test <- read.csv("/Users/souvikbag/Library/CloudStorage/OneDrive-UniversityofMissouri/Assignments/Fall 23/STAT 8330/Project 1/Project 1/test.csv")

library(caret)
library(ggplot2)
library(ISLR2)
library(tidyverse)
library(dplyr)
library(tidyverse)
library(magrittr)

# convert smoking status to a factor 

columns_to_convert <- c("smoking", "hearing.left.", "hearing.right.", "Urine.protein","dental.caries")  # Replace with your column names

train <- train %>% mutate_at(vars(columns_to_convert), as.factor)
train <-  subset(train, select = - c(id))

test <- test %>% mutate_at(vars(setdiff(columns_to_convert,"smoking")), as.factor)
test <-  subset(test, select = - c(id))
## Train Test split
training.sample <- train$smoking %>% createDataPartition(p = 0.99 , list = FALSE) 

train.data <- train[training.sample,]
test.data <- train[-training.sample,]


x.train <- model.matrix(smoking ~.-1, data = train.data) %>% scale()
x.test <- model.matrix(smoking ~.-1, data = test.data) %>% scale()
x.test <- array_reshape(x.test  , c(nrow(x.test ), 27))
x.train<- array_reshape(x.train , c(nrow(x.train ), 27))

y <- data$smoking
y.train <- to_categorical(train.data$smoking,2)
y.test <- to_categorical(test.data$smoking,2)


# preprocess test data for prediction

test <- model.matrix( ~.-1, data = test) %>% scale()
test <- array_reshape(test  , c(nrow(test ), 27))

modnn <- keras_model_sequential () %>%
  layer_dense(units = 20, activation = "relu",
              input_shape = ncol(x.train)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 2, activation = "softmax")

modnn %>% compile(loss = "categorical_crossentropy",
                  optimizer = optimizer_rmsprop (), metrics = c("accuracy"))

history <- modnn %>%
  fit(x.train , y.train , epochs = 30,
      validation_split = 0.2)

modnn %>% predict(test)


as.vector(predicted)

###################

modelnn <- keras_model_sequential ()
modelnn %>%
  layer_dense(units = 50, activation = "relu",
              input_shape = ncol(x.train)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 2, activation = "softmax")

modelnn %>% compile(loss = "categorical_crossentropy",
                    optimizer = optimizer_rmsprop (), metrics = c("accuracy"))


modelnn %>%
  fit(x.train , y.train , epochs = 30,
      validation_split = 0.2)


# prediction and submission 

p <- predict(modelnn, test)
p <- data.frame(p)
Metrics :: auc(train.data$smoking, p[,2])

test_org <- read.csv("/Users/souvikbag/Library/CloudStorage/OneDrive-UniversityofMissouri/Assignments/Fall 23/STAT 8330/Project 1/Project 1/test.csv")

results_submision_nn <- cbind("id" = test_org$id, "smoking" =  p$X2)

write.csv(results_submision_nn, file = "results_sub_nn.csv",row.names = FALSE)










