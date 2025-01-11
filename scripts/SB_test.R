test<-read.csv("test.csv")
test$SalePrice <- rep(0, nrow(test))
# Dealing with the NAs 
test$Alley[is.na(test$Alley)] <- "No"
test$MasVnrType[is.na(test$MasVnrType)] <- "UNKN"
test$MasVnrArea[is.na(test$MasVnrArea)] <- "UNKN"
test$BsmtCond[is.na(test$BsmtCond)] <- "No"
test$BsmtQual[is.na(test$BsmtQual)] <- "No"
test$BsmtExposure[is.na(test$BsmtExposure)] <- "No"
test$BsmtFinType1[is.na(test$BsmtFinType1)] <- "No"
test$BsmtFinType2[is.na(test$BsmtFinType2)] <- "No"
test$FireplaceQu[is.na(test$FireplaceQu)] <- "No"
test$GarageType[is.na(test$GarageType)] <- "No"
test$GarageYrBlt[is.na(test$GarageYrBlt)] <- "No"
test$GarageFinish[is.na(test$GarageFinish)] <- "No"
test$GarageQual[is.na(test$GarageQual)] <- "No"
test$GarageCond[is.na(test$GarageCond)] <- "No"
test$PoolQC[is.na(test$PoolQC)] <- "No"
test$Fence[is.na(test$Fence)] <- "No"
test$MiscFeature[is.na(test$MiscFeature)] <- "No"
test$Electrical[is.na(test$Electrical)] <- "No"

no_order_vars <- c("Street", "MSZoning", "MSSubClass", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood",
                   "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "Exterior1st",
                   "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "Functional","GarageType",
                   "Fence","MiscFeature", "SaleType","SaleCondition", "RoofMatl")


test[no_order_vars] <- lapply(test[no_order_vars], factor)


test$OverallQual <- factor(test$OverallQual, levels = c( "1" , "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10"), ordered = T)
test$OverallCond <- factor(test$OverallCond, levels = c( "1" , "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10"), ordered = T)
test$ExterQual <- factor(test$ExterQual, levels = c("Po","Fa","TA","Gd","Ex"), ordered = T)
test$ExterCond <- factor(test$ExterCond, levels = c("Po","Fa","TA","Gd","Ex"), ordered = T)
test$BsmtQual <- factor(test$BsmtQual, levels = c("No","Po","Fa","TA","Gd","Ex"), ordered = T)
test$BsmtCond <- factor(test$BsmtCond, levels = c("No","Po","Fa","TA","Gd","Ex"), ordered = T)
test$BsmtExposure <- factor(test$BsmtExposure, levels = c("No","Mn","Av","Gd"), ordered = T)
test$BsmtFinType1 <- factor(test$BsmtFinType1, levels = c("No","Unf","LwQ","Rec","BLQ", "ALQ", "GLQ"), ordered = T)
test$BsmtFinType2 <- factor(test$BsmtFinType2, levels = c("No","Unf","LwQ","Rec","BLQ", "ALQ", "GLQ"), ordered = T)
test$HeatingQC <- factor(test$HeatingQC, levels = c("Po","Fa","TA","Gd","Ex"), ordered = T)
test$Electrical <- factor(test$Electrical, levels = c("Mix","FuseP","FuseF","FuseA","SBrkr"), ordered = T)
test$KitchenQual <- factor(test$KitchenQual, levels = c("Po","Fa","TA","Gd","Ex"), ordered = T)
test$FireplaceQu <- factor(test$FireplaceQu, levels = c("No","Po","Fa","TA","Gd","Ex"), ordered = T)
test$GarageFinish <- factor(test$GarageFinish, levels = c("No","Unf","RFn","Fin"), ordered = T)
test$GarageQual <- factor(test$GarageQual, levels = c("No","Po","Fa","TA","Gd","Ex"), ordered = T)
test$GarageCond <- factor(test$GarageCond, levels = c("No","Po","Fa","TA","Gd","Ex"), ordered = T)
test$PavedDrive <- factor(test$PavedDrive, levels = c("N","P","Y"), ordered = T)
test$PoolQC <- factor(test$PoolQC, levels = c("No","Fa","TA","Gd","Ex"), ordered = T)



#RF model for imputation of LotFrontage

testtrain<- test[rowSums(is.na(test)) ==0,]
dim(testtrain)
testtest<- test[rowSums(is.na(test)) >0,]
dim(testtest)

model <- randomForest(LotFrontage ~ ., data = testtrain, importance=TRUE, ntree=3000) 
LFpred_test<-predict(model,testtest[,-c(4)])
test$LotFrontage<-ifelse(is.na(test$LotFrontage)==TRUE,LFpred_test,test$LotFrontage)



# Remove specified columns
df_test <- imputed_data %>%
  select(-one_of(columns_to_remove))
# Remove other NAs

df_test <- na.omit(df_test)

# Identify categorical columns
categorical_cols <- sapply(df, is.factor)

# Identify unique levels for each categorical column in the training dataset
levels_train <- lapply(df[, categorical_cols, drop = FALSE], levels)

# Update levels in the test dataset to match those in the training dataset
for (col_name in names(df)[categorical_cols]) {
  levels_test <- levels(df_test[[col_name]])
  
  if (!identical(levels_train[[col_name]], levels_test)) {
    cat("Updating levels for column", col_name, "\n")
    df_test[[col_name]] <- factor(df_test[[col_name]], levels = levels_train[[col_name]])
  }
}

imputed_data$Utilities <-  ifelse(is.na(imputed_data$Utilities) == TRUE, "NoSewr", imputed_data$Utilities)


# Initialize an empty list to store unique levels for each column
all_levels <- list()

# Use a loop to compute unique levels for each specified column
for (col in different_levels) {
  all_levels[[col]] <- unique(c(levels(df[,col]), levels(imputed_data[,col])))
}

# Make each specified categorical column in both data frames a factor with its levels
for (col in different_levels) {
  levels_to_use <- factor(all_levels[[col]])
  
  df[,col] <- factor(  df[,col] , levels = levels_to_use)
  imputed_data[,col] <- factor(imputed_data[,col], levels = levels_to_use)
}






imputed_data_org <- imputed_data
# df_test <- na.omit(df_test)
# Remove specified columns
imputed_data <- imputed_data %>%
  select(-one_of(columns_to_remove))
# Identify the indices of categorical columns
categorical_indices_test <- sapply(imputed_data, is.factor)

# Perform one-hot encoding only for categorical columns
encoded_df_test <- model.matrix(~ . - 1, data = imputed_data[, categorical_indices_test])
# Combine the original data with the encoded data
final_df_test <- cbind(imputed_data, encoded_df_test)

# Remove the original categorical columns if needed
final_df_test <- final_df_test[, !categorical_indices_test]


which(colnames(final_df_test)=="SalePrice")

# Create a matrix of predictors (excluding the response column)
predictors_matrix_test <- as.matrix(final_df_test[,-34])
# Ensure that the matrix contains numeric data
predictors_matrix_test <- apply(predictors_matrix_test, 2, as.numeric)
# Create a numeric vector for the response
response_vector_test <- as.numeric(final_df_test[,34])



# Convert data to a DMatrix
xgbtest <- xgb.DMatrix(data = predictors_matrix_test, label = response_vector_test)

pred_test <- predict(xgb_model2, newdata = xgbtest)
pred_test2 <- predict(xgb_model2, newdata = xgbtest)
submission1 <- data.frame(Id = test$Id, SalePrice = pred_test)
submission2 <- data.frame(Id = test$Id, SalePrice = pred_test2)

write.csv(submission2, file = "Submission10_xgboost.csv", row.names = F)



