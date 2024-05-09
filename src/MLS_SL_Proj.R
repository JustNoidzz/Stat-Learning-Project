### MLS Soccer ML Project ###
## Patrick Geraghty & Zidong Liu ##

# set working directory 
setwd("~/Lab Dataset")

# import data set 
library(tidyverse)
library(dplyr)
library(readxl)
library(ggplot2)
MLSML <- read_xlsx("Stat Learning Project Data.xlsx")

# *delete games that ended in a draw*
MLSML <- MLSML %>% 
  filter(Result_A != "D")

# make result variable a factor 
#MLSML$Result_A <- gsub("L", "-1", MLSML$Result_A)
#MLSML$Result_A <- gsub("D", "0", MLSML$Result_A)
#MLSML$Result_A <- gsub("W", "1", MLSML$Result_A)
MLSML$Result_A <- as.factor(MLSML$Result_A)

# check distribution of variable (checks out)
summary(MLSML$Result_A)   
plot(MLSML$Result_A)

# delete useless variables
MLSML <- MLSML %>% select(-Season)
MLSML <- MLSML %>% select(-Date)
#MLSML <- MLSML %>% select(-Venue_A)      # comment out for home away models 
MLSML <- MLSML %>% select(-TeamA)
MLSML <- MLSML %>% select(-TeamB)
MLSML <- MLSML %>% select(-Gls_A)
MLSML <- MLSML %>% select(-Gls_B)
MLSML <- MLSML %>% select(-`Goals/Shot_A`)
MLSML <- MLSML %>% select(-`Goals/Shot_B`)
MLSML <- MLSML %>% select(-`Goals/SOT_A`)
MLSML <- MLSML %>% select(-`Goals/SOT_B`)
MLSML <- MLSML %>% select(-`G-xG_A`)
MLSML <- MLSML %>% select(-`G-xG_B`)
MLSML <- MLSML %>% select(-Ast_A)
MLSML <- MLSML %>% select(-Ast_B)

# data normalization
library(caret)
ss <- preProcess(as.data.frame(MLSML), method=c("range"))
MLSML <- predict(ss, as.data.frame(MLSML))

# divide into training and testing set (will use k fold CV on training)
library(boot)
set.seed(1)
index_all <- sample(1:nrow(MLSML), size = nrow(MLSML), replace = FALSE)
index_train <- sample(index_all, size = round(nrow(MLSML) * 0.8))
index_test <- setdiff(index_all, index_train)
data_train <- MLSML[index_train, ]
data_test <- MLSML[-index_train, ]
resulttest <- MLSML$Result_A[-index_train]

# Define the training control for cv
ctrl <- trainControl(method = "cv", number = 5)
# Define the formula for model including what coaches could prioritize
formula <- Result_A ~ PassCmpRate_A + Possession_A + PassPrgDist_A + ShareTouchAttThird_A + 
  FinalThirdEntries_A + TouchesAttThird_A + LongThroughSwitch_TotalPasses_A + Directness_A + 
  Short_TotalPasses_A + PlaySpeed_A + Crosses_A + PenBoxEntries_A + PenEntriesPerAttThirdEntry_A + 
  npxG_Shot_A + Shots_A + PPDA_A + Challenges_A + PassCmpRate_B + ShareTklAttThird_A + 
  ChallengesWonRate_A + TklWonRate_A + AerialWonRate_A + LooseRecovWonRate_A + Directness_B + 
  LongThroughSwitch_TotalPasses_B + Short_TotalPasses_B + ShareTouchAttThird_B + Crosses_B 
# in possession
formulaIP <- Result_A ~ PassCmpRate_A + Possession_A + PassPrgDist_A + ShareTouchAttThird_A + 
  FinalThirdEntries_A + TouchesAttThird_A + LongThroughSwitch_TotalPasses_A + Directness_A + 
  Short_TotalPasses_A + PlaySpeed_A + Crosses_A + PenBoxEntries_A + PenEntriesPerAttThirdEntry_A + 
  npxG_Shot_A + Shots_A
# out of possession
formulaOP <- Result_A ~ PPDA_A + Challenges_A + PassCmpRate_B + ShareTklAttThird_A + 
  ChallengesWonRate_A + TklWonRate_A + AerialWonRate_A + LooseRecovWonRate_A + Directness_B + 
  LongThroughSwitch_TotalPasses_B + Short_TotalPasses_B + ShareTouchAttThird_B + Crosses_B 
# with home/away information 
formulaHA <- Result_A ~ PassCmpRate_A + Possession_A + PassPrgDist_A + ShareTouchAttThird_A + 
  FinalThirdEntries_A + TouchesAttThird_A + LongThroughSwitch_TotalPasses_A + Directness_A + 
  Short_TotalPasses_A + PlaySpeed_A + Crosses_A + PenBoxEntries_A + PenEntriesPerAttThirdEntry_A + 
  npxG_Shot_A + Shots_A + PPDA_A + Challenges_A + PassCmpRate_B + ShareTklAttThird_A + 
  ChallengesWonRate_A + TklWonRate_A + AerialWonRate_A + LooseRecovWonRate_A + Directness_B + 
  LongThroughSwitch_TotalPasses_B + Short_TotalPasses_B + ShareTouchAttThird_B + Crosses_B + Venue_A
# xG
formulaxG <- Result_A ~ PassCmpRate_A + Possession_A + PassPrgDist_A + ShareTouchAttThird_A + 
  FinalThirdEntries_A + TouchesAttThird_A + LongThroughSwitch_TotalPasses_A + Directness_A + 
  Short_TotalPasses_A + PlaySpeed_A + Crosses_A + PenBoxEntries_A + PenEntriesPerAttThirdEntry_A + 
  npxG_Shot_A + Shots_A + PPDA_A + Challenges_A + PassCmpRate_B + ShareTklAttThird_A + 
  ChallengesWonRate_A + TklWonRate_A + AerialWonRate_A + LooseRecovWonRate_A + Directness_B + 
  LongThroughSwitch_TotalPasses_B + Short_TotalPasses_B + ShareTouchAttThird_B + Crosses_B + xDif_A
# subset data for correlation matrix
subs <- subset(MLSML, select=c('PassCmpRate_A', 'Possession_A', 'PassPrgDist_A', 'ShareTouchAttThird_A', 
               'FinalThirdEntries_A', 'TouchesAttThird_A', 'LongThroughSwitch_TotalPasses_A', 'Directness_A', 
               'Short_TotalPasses_A', 'PlaySpeed_A', 'Crosses_A', 'PenBoxEntries_A', 'PenEntriesPerAttThirdEntry_A', 
               'npxG_Shot_A', 'Shots_A', 'PPDA_A', 'Challenges_A', 'PassCmpRate_B', 'ShareTklAttThird_A', 
               'ChallengesWonRate_A', 'TklWonRate_A', 'AerialWonRate_A', 'LooseRecovWonRate_A', 'Directness_B', 
               'LongThroughSwitch_TotalPasses_B', 'Short_TotalPasses_B', 'ShareTouchAttThird_B', 'Crosses_B'))
# Correlation Matrix of Subset Dataset
library(corrplot)
cor <- cor(subs)
corrplot(cor)


#######
# LDA #
#######
library(MASS)
# USING K=5 CV ON TRAINING BEFORE TESTING
# Train LDA model using cross-validation
lda_model <- train(formula, data = data_train, method = "lda", trControl = ctrl)
# View the cross-validation results
lda_model
# Train LDA model on the entire training dataset
final_lda_model <- lda(formula, data = data_train)
final_lda_model
ldapred=predict(final_lda_model, data_test)
ldaclass=ldapred$class
table(ldaclass, data_test$Result_A)
mean(ldaclass==data_test$Result_A)
# Extract coefficients (linear discriminants) from the LDA model
coefficients <- coef(final_lda_model)
# Add row names as a column
df <- data.frame(coefficients)
coefficients <- rownames_to_column(df, var = "row_names")
coefficients <- coefficients %>% rename(Coefficients = LD1)
# Make bar graph of best coefficients
ggplot(data = coefficients, aes(x = reorder(row_names, Coefficients), y = Coefficients,fill=abs(Coefficients))) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_gradient(low="lightblue", high="darkred") + 
  labs(title="Linear Discriminant Model Predictors by Coefficient Weight", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  coord_flip()

# in possession 
lda_model <- train(formulaIP, data = data_train, method = "lda", trControl = ctrl)
lda_model
final_lda_model <- lda(formulaIP, data = data_train)
final_lda_model
ldapred=predict(final_lda_model, data_test)
ldaclass=ldapred$class
table(ldaclass, data_test$Result_A)
mean(ldaclass==data_test$Result_A)
coefficients <- coef(final_lda_model)
df <- data.frame(coefficients)
coefficients <- rownames_to_column(df, var = "row_names")
coefficients <- coefficients %>% rename(Coefficients = LD1)
ggplot(data = coefficients, aes(x = reorder(row_names, Coefficients), y = Coefficients,fill=abs(Coefficients))) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_gradient(low="lightblue", high="darkred") + 
  labs(title="Linear Discriminant Model (In Possession) Predictors by Coefficient Weight", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  coord_flip()

# out of possession 
lda_model <- train(formulaOP, data = data_train, method = "lda", trControl = ctrl)
lda_model
final_lda_model <- lda(formulaOP, data = data_train)
final_lda_model
ldapred3=predict(final_lda_model, data_test)
ldaclass=ldapred$class
table(ldaclass, data_test$Result_A)
mean(ldaclass==data_test$Result_A)
coefficients <- coef(final_lda_model)
df <- data.frame(coefficients)
coefficients <- rownames_to_column(df, var = "row_names")
coefficients <- coefficients %>% rename(Coefficients = LD1)
ggplot(data = coefficients, aes(x = reorder(row_names, Coefficients), y = Coefficients,fill=abs(Coefficients))) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_gradient(low="lightblue", high="darkred") + 
  labs(title="Linear Discriminant Model (Out of Possession) Predictors by Coefficient Weight", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  coord_flip()

# home away 
lda_model <- train(formulaHA, data = data_train, method = "lda", trControl = ctrl)
lda_model
final_lda_model <- lda(formulaHA, data = data_train)
final_lda_model
ldapred=predict(final_lda_model, data_test)
ldaclass=ldapred$class
table(ldaclass, data_test$Result_A)
mean(ldaclass==data_test$Result_A)

# xG 
lda_model <- train(formulaxG, data = data_train, method = "lda", trControl = ctrl)
lda_model
final_lda_model <- lda(formulaxG, data = data_train)
final_lda_model
ldapred=predict(final_lda_model, data_test)
ldaclass=ldapred$class
table(ldaclass, data_test$Result_A)
mean(ldaclass==data_test$Result_A)

#######################
# Logistic Regression #
#######################
# USING K=5 CV ON TRAINING BEFORE TESTING
logmodel <- train(formula, data=data_train, method="glm", trControl=ctrl, family="binomial")
logmodel
# Fit the model on the entire training set
finallogmodel <- glm(formula, data=data_train, family="binomial")
summary(finallogmodel)
# Probabilities and Predictions
glmprobs = predict(finallogmodel, data_test, type="response")
glmpred = rep("L", length(index_test))
glmpred[glmprobs >.5]="W"
# Confusion Matrix
table(glmpred, data_test$Result_A)
# Test Accuracy
mean(glmpred == data_test$Result_A)
# Extract coefficients
coefficients2 <- coef(finallogmodel)
# Add row names as a column
df <- data.frame(coefficients2)
coefficients2 <- rownames_to_column(df, var = "row_names")
coefficients2 <- coefficients2 %>% rename(Coefficients = coefficients2)
# Make bar graph of best predictors by coeffcicient weight
ggplot(data = coefficients2, aes(x = reorder(row_names, Coefficients), y = Coefficients,fill=abs(Coefficients))) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_gradient(low="lightblue", high="darkred") + 
  labs(title="Logistic Regression Model Predictors by Coefficient Weight", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  coord_flip()
# Extract p-values
p_values <- summary(finallogmodel)$coefficients[, "Pr(>|z|)"]
# Create a one-column data set with predictor names as row names
p_values_data <- data.frame(p_values)
row.names(p_values_data) <- row.names(summary(finallogmodel)$coefficients)
# View the one-column data set of p-values with predictor names
print(p_values_data)
p_values_data <- rownames_to_column(p_values_data, var = "row_names")
p_values_data$statsig <- ifelse(p_values_data$p_values <= .05, "Y", "N")
p_values_data <- p_values_data %>% rename(`1-pvalue` = statsig)
# Make bar graph of stat significant predictors
ggplot(data = p_values_data, aes(x = reorder(row_names, 1-p_values), y = 1-p_values,fill=`1-pvalue`)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values=c("Y"="green","N"="red")) +
  labs(title="Logistic Regression - Statistically Significant Predictors", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  geom_hline(yintercept = .95, linetype = "dashed", color = "black") +
  coord_flip()


# in possession
logmodel <- train(formulaIP, data=data_train, method="glm", trControl=ctrl, family="binomial")
logmodel
finallogmodel <- glm(formulaIP, data=data_train, family="binomial")
glmprobs = predict(finallogmodel, data_test, type="response")
glmpred = rep("L", length(index_test))
glmpred[glmprobs >.5]="W"
table(glmpred, data_test$Result_A)
mean(glmpred == data_test$Result_A)
coefficients2 <- coef(finallogmodel)
df <- data.frame(coefficients2)
coefficients2 <- rownames_to_column(df, var = "row_names")
coefficients2 <- coefficients2 %>% rename(Coefficients = coefficients2)
ggplot(data = coefficients2, aes(x = reorder(row_names, Coefficients), y = Coefficients,fill=abs(Coefficients))) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_gradient(low="lightblue", high="darkred") + 
  labs(title="Best Predictors", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  coord_flip()
p_values <- summary(finallogmodel)$coefficients[, "Pr(>|z|)"]
p_values_data <- data.frame(p_values)
row.names(p_values_data) <- row.names(summary(finallogmodel)$coefficients)
print(p_values_data)
p_values_data <- rownames_to_column(p_values_data, var = "row_names")
p_values_data$statsig <- ifelse(p_values_data$p_values <= .05, "Y", "N")
p_values_data <- p_values_data %>% rename(`1-pvalue` = statsig)
ggplot(data = p_values_data, aes(x = reorder(row_names, 1-p_values), y = 1-p_values,fill=`1-pvalue`)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values=c("Y"="green","N"="red")) +
  labs(title="Logistic Regression - (In Possession) Statistically Significant Predictors", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  geom_hline(yintercept = .95, linetype = "dashed", color = "black") +
  coord_flip()


# out of possession
logmodel <- train(formulaOP, data=data_train, method="glm", trControl=ctrl, family="binomial")
logmodel
finallogmodel <- glm(formulaOP, data=data_train, family="binomial")
glmprobs = predict(finallogmodel, data_test, type="response")
glmpred = rep("L", length(index_test))
glmpred[glmprobs >.5]="W"
table(glmpred, data_test$Result_A)
mean(glmpred == data_test$Result_A)
coefficients2 <- coef(finallogmodel)
df <- data.frame(coefficients2)
coefficients2 <- rownames_to_column(df, var = "row_names")
coefficients2 <- coefficients2 %>% rename(Coefficients = coefficients2)
ggplot(data = coefficients2, aes(x = reorder(row_names, Coefficients), y = Coefficients,fill=abs(Coefficients))) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_gradient(low="lightblue", high="darkred") + 
  labs(title="Best Predictors", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  coord_flip()
p_values <- summary(finallogmodel)$coefficients[, "Pr(>|z|)"]
p_values_data <- data.frame(p_values)
row.names(p_values_data) <- row.names(summary(finallogmodel)$coefficients)
print(p_values_data)
p_values_data <- rownames_to_column(p_values_data, var = "row_names")
p_values_data$statsig <- ifelse(p_values_data$p_values <= .05, "Y", "N")
p_values_data <- p_values_data %>% rename(`1-pvalue` = statsig)
ggplot(data = p_values_data, aes(x = reorder(row_names, 1-p_values), y = 1-p_values,fill=`1-pvalue`)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values=c("Y"="green","N"="red")) +
  labs(title="Logistic Regression - (Out of Possession) Statistically Significant Predictors", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  geom_hline(yintercept = .95, linetype = "dashed", color = "black") +
  coord_flip()

# home away 
logmodel <- train(formulaHA, data=data_train, method="glm", trControl=ctrl, family="binomial")
logmodel
finallogmodel <- glm(formulaHA, data=data_train, family="binomial")
glmprobs = predict(finallogmodel, data_test, type="response")
glmpred = rep("L", length(index_test))
glmpred[glmprobs >.5]="W"
table(glmpred, data_test$Result_A)
mean(glmpred == data_test$Result_A)

# xG
logmodel <- train(formulaxG, data=data_train, method="glm", trControl=ctrl, family="binomial")
logmodel
finallogmodel <- glm(formulaxG, data=data_train, family="binomial")
glmprobs = predict(finallogmodel, data_test, type="response")
glmpred = rep("L", length(index_test))
glmpred[glmprobs >.5]="W"
table(glmpred, data_test$Result_A)
mean(glmpred == data_test$Result_A)

###########
# Bagging #
###########
library(randomForest)
Bagging1=randomForest(formula,data=MLSML,subset=index_train,mtry=28,importance=T)
Bagging1
# predict on test
yhatrf = predict(Bagging1, newdata=data_test, type="class")
confmatrix <- table(yhatrf,resulttest)      
confmatrix
(confmatrix[1,1] + confmatrix[2,2])/282
# importance(): view the importance of each variable
# %IncMSE: mean decrease of accuracy in predictions on the OOB samples when a 
# given variable is excluded from the model
# IncNodeImpurity: total decrease in node impurity that results from splits over
# that variable, averaged over all trees (RSS in regr. vs. deviance in class)
importance(Bagging1)
# varImpPlot(): Variance importance plot
varImpPlot(Bagging1)

# with all variables
Bagging2=randomForest(Result_A~.-xDif_A-xG_A-xG_B-npxG_A-npxG_B-xAG_A-xAG_B,
                     data=MLSML,subset=index_train,mtry=180,importance=T)
Bagging2
yhatrf = predict(Bagging2, newdata=data_test, type="class")
confmatrix <- table(yhatrf,resulttest)      
confmatrix
(confmatrix[1,1] + confmatrix[2,2])/282
importance(Bagging2)
varImpPlot(Bagging2)

##################
# Random Forests #
##################
# By default randomForest() uses m=p/3 for regression and m=sqrt(p) for classification
RandomForests1=randomForest(formula,data=MLSML,subset=index_train,mtry=14,importance=T)
RandomForests1
yhatrf = predict(RandomForests1, newdata=data_test, type="class")
confmatrix <- table(yhatrf,resulttest)      
confmatrix
(confmatrix[1,1] + confmatrix[2,2])/282
importance(RandomForests1)
varImpPlot(RandomForests1)

RandomForests2=randomForest(Result_A~.-xDif_A-xG_A-xG_B-npxG_A-npxG_B-xAG_A-xAG_B,
                    data=MLSML,subset=index_train,mtry=90,importance=T)
RandomForests2
yhatrf = predict(RandomForests2, newdata=data_test, type="class")
confmatrix <- table(yhatrf,resulttest)      
confmatrix
(confmatrix[1,1] + confmatrix[2,2])/282
importance(RandomForests2)
varImpPlot(RandomForests2)





#################
# Brainstorming #
#################
# Exploratory Data Analysis #
library(ggplot2)
# histograms of potential predictors based on result category
ggplot(MLSML, aes(x=Clr_A, fill=Result_A)) +
  geom_histogram() +
  scale_fill_manual(values=c("W"="green","L"="red")) +
  labs(title = "Results by Clearances") +
  theme_minimal()
cor(MLSML$Clr_A,MLSML$xDif_A)
cor(MLSML$Clr_A,MLSML$xG_A)
cor(MLSML$Clr_A,MLSML$xG_B)
ggplot(MLSML, aes(x=FieldTilt_A, fill=Result_A)) +
  geom_histogram() +
  scale_fill_manual(values=c("W"="green","L"="red")) +
  labs(title = "Results by Field Tilt") +
  theme_minimal()
ggplot(MLSML, aes(x=PPDA_A, fill=Result_A)) +
  geom_histogram() +
  scale_fill_manual(values=c("W"="green","L"="red")) +
  labs(title = "Results by PPDA") +
  theme_minimal()
ggplot(MLSML, aes(x=Directness_A, fill=Result_A)) +
  geom_histogram() +
  scale_fill_manual(values=c("W"="green","L"="red")) +
  labs(title = "Results by Directness") +
  stat_bin(bins = 30) +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) 
ggplot(MLSML, aes(x=PlaySpeed_A, fill=Result_A)) +
  geom_histogram() +
  scale_fill_manual(values=c("W"="green","L"="red")) +
  labs(title = "Results by Speed of Play in Possession") +
  stat_bin(bins = 30) +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) 
cor(MLSML$xG_B,MLSML$PlaySpeed_A)
ggplot(MLSML, aes(x=Crosses_A,y=xG_A,color=Result_A)) + 
  geom_point() +
  scale_color_manual(values=c("W"="green","L"="red")) +
  labs(title = "Relationship Between Total Cross Count and Expected Goals", x = "Crosses", y = "xG") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
      panel.grid.major = element_blank(), panel.grid.minor = element_blank()) 
summary(MLSML$Crosses_A)
cross <- MLSML %>% filter(Crosses_A >= 17)
nocross <- MLSML %>% filter(Crosses_A <= 17)
summary(cross$Result_A)
summary(nocross$Result_A)
cor(cross$Crosses_A,cross$xG_A)
cor(nocross$Crosses_A,nocross$xG_A)

cor(MLSML$npxG_Shot_A,MLSML$Gls_A)




# what is win % when a team does xyz.....
summary(MLSML$Possession_A)
summary(MLSML$PPDA_A)
summary(MLSML$FieldTilt_A)
summary(MLSML$Directness_A)
BB1 <- MLSML %>% filter(PPDA_A <= 7.5 | Directness_A >= .4 | FieldTilt_A >= .6)
summary(BB1$Result_A)

# giving up possession is fine if..... you keep them out of the final third
summary(MLSML$ShareTouchAttThird_B)
summary(MLSML$Possession_A)
BB2 <- MLSML %>% filter(Possession_B >= 55 | ShareTouchAttThird_B <= .2)
cor(BB2$Gls_B,BB2$Possession_B)
cor(MLSML$Gls_B,MLSML$Possession_B)

# does being more dependent on crosses to get into box affect xG
MLSML$CrossDependency_A<-MLSML$SuccCross_A/MLSML$SuccPassIntoBox_A
correlation <- cor(MLSML$Gls_A, MLSML$CrossDependency_A, use = "complete.obs") #correlation=-.14

