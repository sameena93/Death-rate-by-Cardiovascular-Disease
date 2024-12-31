#clear the environment
rm(list=ls())


#Import the libraries
library(ggplot2)
library(dplyr)
library(tidyverse)
library(ggcorrplot)
library(explore)
library(gridExtra)
library(caret)
# library(MASS)
# install.packages("pROC")
library(pROC)
library(e1071)
# library(nnet)


#Load the dataset
heart_failure_dataset <- read.csv("C:/Users/moham/DataScience/Predictive modeling/PM_ projects/Final Project/heart_failure_dataset.csv")
View(heart_failure_dataset)

#Display the first few rows
head(heart_failure_dataset)

#last rows of the datset
tail(heart_failure_dataset)

#Column names 
colnames(heart_failure_dataset)

#summary of dataset
summary(heart_failure_dataset)

#structure of dataset
str(heart_failure_dataset)

#check for missing values
sum(is.na(heart_failure_dataset))


column_classes <- sapply(heart_failure_dataset,class)
print(column_classes)


# 
# convert_to_numerical <- function(data) {
#   for (col in names(data)) {
#     if (is.factor(data[[col]]) || is.character(data[[col]])) {
#       print(paste("Column:", col, "Type:", class(data[[col]])))
#       print(unique(data[[col]]))
#       
#       if (nlevels(data[[col]]) == 2) {
#         # Binary classification, convert to binary numeric variable
#         binary_level <- levels(data[[col]])[2]
#         data[[col]] <- as.numeric(data[[col]] == binary_level)
#       # } else if (nlevels(data[[col]]) > 2) {
#       #   # One-hot encoding for more than two levels
#       #   encoded_col <- model.matrix(~ data[[col]] - 1)
#       #   colnames(encoded_col) <- paste(col, colnames(encoded_col), sep='_')
#       #   
#       #   # Replace original column with one-hot encoded column
#       #   data[[col]] <- as.numeric(encoded_col[, 1])
#       }
#     }
#   }
#   return(data)
# }
convert_categorical_to_numeric <- function(data, column_name) {
  # Check if the column is binary
  if (length(unique(data[[column_name]])) == 2) {
    # If binary, convert to 0 and 1
    data[[column_name]] <- as.numeric(factor(data[[column_name]], levels = levels(factor(data[[column_name]]))))
  } else {
    # If more than two levels, use one-hot encoding
    encoded <- model.matrix(~ data[[column_name]] - 1)
    colnames(encoded) <- gsub("data[[column_name]]", "", colnames(encoded))
    data <- cbind(data, encoded)
    # Remove the original categorical column
    data[[column_name]] <- NULL
  }
  return(data)
}

# Example usage:
# Assuming you have a data frame named 'your_data' with a categorical column named 'your_categorical_column'
your_data <- convert_categorical_to_numeric(your_data, 'your_categorical_column')


convert_categorical_to_numeric(heart_failure_dataset)


head(heart_failure_dataset)
  
length(unique(heart_failure_dataset$anaemia))
levels(heart_failure_dataset$diabetes)
unique(heart_failure_dataset$diabetes)
# converting the categorical column to as a factor
heart_failure_dataset$anaemia <- as.factor(heart_failure_dataset$anaemia)
heart_failure_dataset$diabetes <- as.factor(heart_failure_dataset$diabetes)
heart_failure_dataset$high_blood_pressure <- as.factor(heart_failure_dataset$high_blood_pressure )
heart_failure_dataset$smoking <- as.factor(heart_failure_dataset$smoking)
heart_failure_dataset$gender <- as.factor(heart_failure_dataset$gender)
heart_failure_dataset$DEATH_EVENT <- as.factor(heart_failure_dataset$DEATH_EVENT)



# Divide the categorical and numerical columns

## Create an empty lists
cat_col <- character(0)
num_col <- character(0)

for (col in colnames(heart_failure_dataset)) {
  if (is.factor(heart_failure_dataset[[col]])) {
    cat_col <- c(cat_col, col)
  } else if (is.numeric(heart_failure_dataset[[col]])) {
    num_col <- c(num_col, col)
  }
}

#Display the categorical and numerical columns lists
cat_col
num_col  




#Exploratory data analysis

heart_failure_dataset %>% 
  ggplot(aes(x=smoking, fill=gender))+
  geom_bar(position= position_dodge())


heart_failure_dataset %>%
  ggplot(aes(x = age, y = ejection_fraction, color = DEATH_EVENT)) +
  geom_point(aes(shape = DEATH_EVENT), size = 2) +
  labs(title = "Ejection Fraction Vs Age by Death Event", x = "Age", y = "Ejection Fraction")+
  theme(plot.title = element_text(face= 'bold', hjust=0.5, size=18),
        axis.title = element_text(face='bold', size=14))+
  scale_color_manual(values = c('Yes' = "red", 'No' ="#4DBEEE"))
  
ggsave("C:\\Users\\moham\\DataScience\\Predictive modeling\\PM_ projects\\Final Project\\ejection_fraction_by_age.jpeg", width=10, height=8)



#Density plot of hospital visit by death events
heart_failure_dataset %>%  
ggplot(aes(x=time, fill=DEATH_EVENT))+
  geom_density()+
  scale_fill_manual(values = c('Yes'='#F7F4BF', 'No'='#FCEAE6'))+
  labs(title= "Death Events by Number of Visits to the Hospital", x="Time(Number of visit)", y="Density")+
  theme(plot.title = element_text(face= 'bold', hjust=0.5),
        axis.title = element_text(face='bold',size=12))


ggsave("C:\\Users\\moham\\DataScience\\Predictive modeling\\PM_ projects\\Final Project\\ejection fraction by age.jpeg", width=8, height=3)


#Explore graph using all variables
heart_failure_dataset %>% explore_all(target=DEATH_EVENT)



#Histogram of serum creatinine by death events with high blood pressure

heart_failure_dataset %>% 
  ggplot(aes(x=serum_creatinine, fill=high_blood_pressure))+
  geom_histogram(color= 'black', bins=30)+
  facet_wrap(~ DEATH_EVENT)+
  scale_fill_manual(values= c('No' = 'mistyrose', 'Yes'= 'plum3'))+
  labs(title= 'Histogram of Sr Creatinine by Death Events with HTN',
       x= 'Serum Creatinine',
       y= 'Count')+
  theme(plot.title = element_text(face= 'bold', hjust=0.5),
        axis.title = element_text(face='bold',size=12))

ggsave("C:\\Users\\moham\\DataScience\\Predictive modeling\\PM_ projects\\Final Project\\creatbyhtn.jpeg", width=6, height=4)


# scatter plot of serum creatinine by creatinine_phosphokinase filled with death events
heart_failure_dataset %>% 
ggplot(aes(x = serum_creatinine,y= creatinine_phosphokinase, color = DEATH_EVENT)) +
  geom_point() +
  labs(
    title = "Creatinine vs. creatinine_phosphokinase by Death Event",
    x = 'Serum Creatinine',
    y = " Creatinine Phosphokinase",
    color = "Death Event"
  ) +
  theme(plot.title = element_text(hjust=0.5, face='bold'),
        axis.title =  element_text(face='bold'),
        legend.title  = element_text((face='bold')))



a<- unique(heart_failure_dataset$anaemia)
print(a)
print(a[2])

b<- model.matrix(~heart_failure_dataset$anaemia -1)
b
# converting the categorical column to numeric column
heart_failure_dataset$anaemia <- as.numeric(heart_failure_dataset$anaemia == "Yes")
heart_failure_dataset$diabetes <- as.numeric(heart_failure_dataset$diabetes == 'Yes')
heart_failure_dataset$high_blood_pressure <- as.numeric(heart_failure_dataset$high_blood_pressure == "Yes")
heart_failure_dataset$smoking <- as.numeric(heart_failure_dataset$smoking == "Yes")
heart_failure_dataset$gender <- as.numeric(heart_failure_dataset$gender == 'Male')
heart_failure_dataset$DEATH_EVENT <- as.factor(heart_failure_dataset$DEATH_EVENT)

# Seprate the numerical columns 
numeric_variable <- sapply(heart_failure_dataset, is.numeric)
numeric_col <- heart_failure_dataset[, numeric_variable]
colnames(numeric_col)




# #Standardize the numerical variables
# std_num <- scale(heart_failure_dataset[num_col])
# 
# std_num



#calculate the correlation of matrix
cor_matrix <- cor(numeric_col)

ggcorrplot(cor_matrix, hc.order = TRUE, lab=TRUE, 
           title= "Heatmap of Coorelation Matrix")+
  theme(plot.title= element_text(face='bold', hjust=0.5, size=20),
        axis.text = element_text(face='bold', size=12),
        legend.title = element_text(face="bold"),
        legend.text =element_text(face="bold"))


# saving the heatmap
ggsave("C:\\Users\\moham\\DataScience\\Predictive modeling\\PM_ projects\\Final Project\\heatmap of coorelation matrix.jpeg", 
       width = 12, height=12, limitsize = FALSE)



eigen(cor_matrix)

length(unique(heart_failure_dataset$anaemia))

#Prepare the model 
 ## Split the data set

set.seed(123)
training_set <- createDataPartition(heart_failure_dataset$DEATH_EVENT, p=0.70, list=FALSE)

#70% of training dset
training_data <- heart_failure_dataset[training_set, ]
dim(training_data)

# 30% of test data
test_data <- heart_failure_dataset[-training_set, ]
dim(test_data)




# 
# # Combine numerical and categorical columns in the full dataset
# full_data <- cbind(std_num, heart_failure_dataset[cat_col])
# 
# # Split the combined dataset into training and test sets
# set.seed(123)
# training_set <- createDataPartition(full_data$DEATH_EVENT, p = 0.70, list = FALSE)
# 
# # 70% of training dataset
# training_data <- full_data[training_set, ]
# 
# # 30% of test data
# test_data <- full_data[-training_set, ]













# logistic regression model
logistic_model <- train(DEATH_EVENT ~ .,
                        data= training_data,
                        method = 'glm',
                        family= binomial(link= 'logit'))
summary(logistic_model)

#Predict on the test dataset
log_pred <- predict(logistic_model, newdata = test_data)

# Create confusion matrix for logistic model
log_cm<- confusionMatrix(log_pred, test_data$DEATH_EVENT)
log_cm$table


print("Accuracy:")
log_accuracy<- log_cm$overall["Accuracy"]
print(log_accuracy)

print("Precision:")
print(log_cm$byClass["Precision"])

print("Recall (Sensitivity):")
print(log_cm$byClass["Sensitivity"])

print("F1-Score:")
print(log_cm$byClass["F1"])


roc_lr <- roc(response = as.numeric(test_data$DEATH_EVENT), predictor= as.numeric(log_pred))
auc_lr <- auc(roc_lr)

print(auc_lr)

roc_lr <- roc(as.numeric(log_pred), as.numeric(test_data$DEATH_EVENT))
print("AUC(Area Under the ROC Curve):")
print(auc(roc_lr))

plot(roc_lr, main = "ROC Curve", col = "blue", lwd = 2)





## Decision Tree Model
# Define the hyperparameter grid
param_grid <- expand.grid(cp = seq(0.01, 0.5, by = 0.01))

# Create a train control object for cross-validation
ctrl <- trainControl(method = "cv", number = 5)

# Perform hyperparameter tuning using caret's train function

dt_model <- train(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time +smoking + high_blood_pressure,
                   data = training_data,
                   method = "rpart", # rpart classification and regression trees
                   tuneGrid = param_grid,
                   trControl = ctrl)

dt_model
# Print the best model and its parameters
dt_model$best_pred

class(heart_failure_dataset$DEATH_EVENT)
#check the model on test set
decision_predict <- predict(dt_model, test_data)
dt_cm <- confusionMatrix(decision_predict, test_data$DEATH_EVENT)
dt_cm$table


print("Accuracy:")
print(dt_cm$overall["Accuracy"])

print("Precision:")
print(dt_cm$byClass["Precision"])

print("Recall (Sensitivity):")
print(dt_cm$byClass["Sensitivity"])

print("F1-Score:")
print(dt_cm$byClass["F1"])


roc_dt <- roc(as.numeric(decision_predict), as.numeric(test_data$DEATH_EVENT))
print("AUC(Area Under the ROC Curve):")
print(auc(roc_dt))

plot(roc_dt, main = "ROC Curve", col = "blue", lwd = 2)






# Create a grid of hyperparameters for each model

rf_grid <- expand.grid(mtry = c(2, 4, 6), nodesize = c(1, 5, 10))

param_grid <- expand.grid(.mtry =c(1,2,3,4,5))



rf_model <- train(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time + smoking,
                  data = training_data,
                  method = "rf", 
                  tuneGrid = param_grid,
                  ntree = 500,
                  trControl = ctrl)
rf_model


# modelrf <- randomForest(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time,
#                         data = training_data)
# plot(modelrf)

rf_predic <- predict(rf_model, test_data)
length(rf_predic)
rf_cm <- confusionMatrix(rf_predic, test_data$DEATH_EVENT)
rf_cm$table

print("Accuracy:")
print(rf_cm$overall["Accuracy"])

print("Precision:")
print(rf_cm$byClass["Precision"])

print("Recall (Sensitivity):")
print(rf_cm$byClass["Sensitivity"])

print("F1-Score:")
print(rf_cm$byClass["F1"])


heart_failure_dataset$DEATH_EVENT <- as.numeric(heart_failure_dataset$DEATH_EVENT == 'Yes')

roc_rf <- roc(as.numeric(rf_predic), as.numeric(test_data$DEATH_EVENT))
print("AUC(Area Under the ROC Curve):")
print(auc(roc_rf))

plot(roc_rf, main = "ROC Curve", col = "blue", lwd = 2)







# Support Vector machine

svm_grid <- expand.grid(
  sigma = c(0.1, 1, 10),  # Specify a range of sigma values
  C = c(0.1, 1, 10)      # Specify a range of C values
)
# Create a train control object for cross-validation
ctrl <- trainControl(method = "cv", number = 5)

svm_model <- train(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time + smoking, 
                   data = training_data,
                   method = "svmRadial", 
                  tuneGrid = svm_grid,
                  trControl = ctrl)


svm_predict <- predict(svm_model, newdata=test_data)
svm_cm <- confusionMatrix(svm_predict,test_data$DEATH_EVENT )
svm_cm$table

print("Accuracy:")
print(svm_cm$overall["Accuracy"])

print("Precision:")
print(svm_cm$byClass["Precision"])

print("Recall (Sensitivity):")
print(svm_cm$byClass["Sensitivity"])

print("F1-Score:")
print(svm_cm$byClass["F1"])

library(pROC) 

# write.csv(svm_cm, file = "svm_confusion_matrix.csv")


#Create the ROC curve and calculate the AUC
roc_svm <- roc(as.numeric(svm_predict), as.numeric(test_data$DEATH_EVENT))
print("AUC(Area Under the ROC Curve):")
print(auc(roc_svm))

plot(roc_svm, main = "ROC Curve", col = "blue", lwd = 2)




#Create a data Frame for comparison of accuracy,Precision, F1 score of all three models

df = data.frame(
  "Models" = c("Logistic Regression","Decision Tree CV",  "Random Forest CV", "Support Vector Machine CV"),
  "Accuracy" = c(log_cm$overall['Accuracy'], dt_cm$overall['Accuracy'], rf_cm$overall["Accuracy"], svm_cm$overall["Accuracy"]),
  "Precision" = c(log_cm$byClass['Precision'], dt_cm$byClass['Precision'], rf_cm$byClass['Precision'], svm_cm$byClass['Precision']),
  "Recall" = c(log_cm$byClass['Recall'], dt_cm$byClass['Recall'],rf_cm$byClass['Recall'], svm_cm$byClass['Recall']),
  "F1 Score" = c(log_cm$byClass['F1'], rf_cm$byClass['F1'],rf_cm$byClass['F1'], svm_cm$byClass['F1'])
)

print(df)
write.table(df, file= "compare the metrics.csv")



library(reshape2)
# Reshape the data for plotting
# Select only the Accuracy and Recall columns
df_selected <- df[, c("Models", "Accuracy", "Recall")]

# Reshape the data for plotting
df_melted <- melt(df_selected, id.vars = "Models")

# Create a grouped bar plot
p <- ggplot(df_melted, aes(x = Models, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(value, 2)), vjust = -0.5), position = position_dodge(width = 0.9)) +
  labs(title = "Model Comparison (Accuracy and Recall)",
       x = "Models",
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Show the plot
print(p)




library(knitr)

results <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"),
  Accuracy = c(log_accuracy, dt_cm$overall["Accuracy"], rf_cm$overall["Accuracy"], svm_cm$overall["Accuracy"]),
  Precision = c(log_cm$byClass["Precision"], dt_cm$byClass["Precision"], rf_cm$byClass["Precision"], svm_cm$byClass["Precision"]),
  Recall = c(log_cm$byClass["Sensitivity"], dt_cm$byClass["Sensitivity"], rf_cm$byClass["Sensitivity"], svm_cm$byClass["Sensitivity"]),
  F1_Score = c(log_cm$byClass["F1"], dt_cm$byClass["F1"], rf_cm$byClass["F1"], svm_cm$byClass["F1"]),
  AUC = c(auc(roc_lr), auc(roc_dt), auc(roc_rf), auc(roc_svm))
)

kable(results, format = "markdown")

# Set the seed for reproducibility
set.seed(123)

# Split the data into a training set and a test set
training_set <- createDataPartition(heart_failure_dataset$DEATH_EVENT, p = 0.70, list = FALSE)
training_data <- heart_failure_dataset[training_set, ]
test_data <- heart_failure_dataset[-training_set, ]

# Logistic Regression Model
logistic_model <- train(DEATH_EVENT ~ ., data = training_data, method = 'glm', family = binomial(link = 'logit'))
log_pred <- predict(logistic_model, newdata = test_data)
log_cm <- confusionMatrix(log_pred, test_data$DEATH_EVENT)
log_accuracy <- log_cm$overall["Accuracy"]

# Decision Tree Model
param_grid_dt <- expand.grid(cp = seq(0.01, 0.5, by = 0.01))
dt_model <- train(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time + smoking,
                  data = training_data, method = "rpart",
                  tuneGrid = param_grid_dt, trControl = ctrl)
decision_predict_dt <- predict(dt_model, test_data)
dt_cm <- confusionMatrix(decision_predict_dt, test_data$DEATH_EVENT)
dt_accuracy <- dt_cm$overall["Accuracy"]

# Random Forest Model
param_grid_rf <- expand.grid(.mtry = c(1, 2, 3, 4, 5))
rf_model <- train(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time + smoking,
                  data = training_data, method = "rf",
                  tuneGrid = param_grid_rf, ntree = 500, trControl = ctrl)
rf_predic <- predict(rf_model, test_data)
rf_cm <- confusionMatrix(rf_predic, test_data$DEATH_EVENT)
rf_accuracy <- rf_cm$overall["Accuracy"]

# Support Vector Machine Model
param_grid_svm <- expand.grid(sigma = c(0.1, 1, 10), C = c(0.1, 1, 10))
svm_model <- train(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time + smoking,
                   data = training_data, method = "svmRadial",
                   tuneGrid = param_grid_svm, trControl = ctrl)
svm_predict <- predict(svm_model, newdata = test_data)
svm_cm <- confusionMatrix(svm_predict, test_data$DEATH_EVENT)
svm_accuracy <- svm_cm$overall["Accuracy"]

# AUC calculation
roc_lr <- roc(as.numeric(log_pred), as.numeric(test_data$DEATH_EVENT))
roc_dt <- roc(as.numeric(decision_predict_dt), as.numeric(test_data$DEATH_EVENT))
roc_rf <- roc(as.numeric(rf_predic), as.numeric(test_data$DEATH_EVENT))
roc_svm <- roc(as.numeric(svm_predict), as.numeric(test_data$DEATH_EVENT))

# Display the results
results <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"),
  Accuracy = c(log_accuracy, dt_accuracy, rf_accuracy, svm_accuracy)
)
print(results)

# Plot the ROC curves
par(mfrow = c(2, 2))
plot(roc_lr, main = "Logistic Regression ROC", col = "blue", lwd = 2)
plot(roc_dt, main = "Decision Tree ROC", col = "blue", lwd = 2)
plot(roc_rf, main = "Random Forest ROC", col = "blue", lwd = 2)
plot(roc_svm, main = "SVM ROC", col = "blue", lwd = 2)



library(pROC)



# Plot the ROC curves
par(mfrow = c(2, 2))
plot(roc_lr, main = "Logistic Regression ROC", col = "blue", lwd = 3,
     print.auc = TRUE, print.auc.coords = c(0.6, 0.4))
plot(roc_dt, main = "Decision Tree ROC", col = "blue", lwd = 3,
     print.auc = TRUE, print.auc.coords = c(0.6, 0.4))
plot(roc_rf, main = "Random Forest ROC", col = "blue", lwd = 3,
     print.auc = TRUE, print.auc.coords = c(0.6, 0.4))
plot(roc_svm, main = "SVM ROC", col = "blue", lwd = 3,
     print.auc = TRUE, print.auc.coords = c(0.6, 0.4))

# Save the plots
dev.copy(png, "C:\\Users\\moham\\DataScience\\Predictive modeling\\PM_ projects\\Final Project\\ROC_plots.png")
dev.off()

