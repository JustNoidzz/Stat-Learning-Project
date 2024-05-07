### MLS Soccer ML Project ###
## Patrick Geraghty & Zidong Liu ##


setwd(getwd())
# import data set 
library(tidyverse)
library(dplyr)
library(readxl)
library(ggplot2)
library(caret)
library(boot)
library(conflicted)
library(MASS)
library(geometry)
conflict_prefer("select", "dplyr")
conflicted::conflicts_prefer(dplyr::filter)


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
#MLSML <- MLSML %>% select(-Venue_A)
#MLSML <- MLSML %>% select(-TeamA)
#MLSML <- MLSML %>% select(-TeamB)
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


#### SWAP ####
# Let TeamA always be the home team

team_a_columns <- grep("_A$", names(MLSML), value = TRUE)
team_b_columns <- grep("_B$", names(MLSML), value = TRUE)
team_a_columns <- setdiff(team_a_columns,c("Venue_A", "Result_A", "xDif_A"))

# Identify when Venue_A == away
away_indices <- which(MLSML$Venue_A == "Away")

# SWAP columns for AWAY DATA
for (i in seq_along(team_a_columns)) {
  
  temp <- MLSML[away_indices, team_a_columns[i]]
  
  MLSML[away_indices, team_a_columns[i]] <- MLSML[away_indices, team_b_columns[i]]
  
  MLSML[away_indices, team_b_columns[i]] <- temp
}

temp_teams <- MLSML$TeamA[away_indices]
MLSML$TeamA[away_indices] <- MLSML$TeamB[away_indices]
MLSML$TeamB[away_indices] <- temp_teams

MLSML$Venue_A[away_indices] <- 'Home'

MLSML$Result_A[away_indices] <- ifelse(MLSML$Result_A[away_indices] == "W", "L", "W")






# data normalization
ss <- preProcess(as.data.frame(MLSML), method=c("range"))
MLSML <- predict(ss, as.data.frame(MLSML))

# divide into training and testing set (will use k fold CV on training)
set.seed(1)
index_all <- sample(1:nrow(MLSML), size = nrow(MLSML), replace = FALSE)
index_train <- sample(index_all, size = round(nrow(MLSML) * 0.8))
index_test <- setdiff(index_all, index_train)
data_train <- MLSML[index_train, ]
data_test <- MLSML[-index_train, ]
resulttest <- MLSML$Result_A[-index_train]




ctrl <- trainControl(method = "cv", number = 5)

# HOME TEAM

formula1 <- Result_A ~ PassCmpRate_A + Possession_A + PassPrgDist_A + ShareTouchAttThird_A + 
  FinalThirdEntries_A + TouchesAttThird_A + LongThroughSwitch_TotalPasses_A + Directness_A + 
  Crosses_A + PenBoxEntries_A + PenEntriesPerAttThirdEntry_A + npxG_Shot_A + Shots_A + 
  PPDA_A + Challenges_A + ShareTklAttThird_A + ChallengesWonRate_A + TklWonRate_A + 
  AerialWonRate_A + LooseRecovWonRate_A + Directness_B + LongThroughSwitch_TotalPasses_B +
  ShareTouchAttThird_B

# USING K=5 CV ON TRAINING BEFORE TESTING
# Train LDA model using cross-validation
lda_model <- train(formula1, data = data_train, method = "lda", trControl = ctrl)
# View the cross-validation results
print(lda_model)
# Train LDA model on the entire training dataset
final_lda_model <- lda(formula1, data = data_train)
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
ggplot(data = coefficients, aes(x = reorder(row_names, Coefficients), y = Coefficients,fill=Coefficients)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_gradient(low="lightblue", high="darkred") + 
  labs(title="Best Predictors HOME TEAM", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  coord_flip()


df$Variable <- rownames(df)

# top_variables
top_variables <- df %>%
  mutate(Absolute_Coefficient = abs(df$LD1)) %>%
  arrange(desc(Absolute_Coefficient)) %>%
  slice(1:5) %>%
  pull(Variable)





# AWAY TEAM
formula2 <- Result_A ~ PassCmpRate_B + Possession_B + PassPrgDist_B + ShareTouchAttThird_B + 
  FinalThirdEntries_B + TouchesAttThird_B + LongThroughSwitch_TotalPasses_B + Directness_B + 
  Crosses_B + PenBoxEntries_B + PenEntriesPerAttThirdEntry_B + npxG_Shot_B + Shots_B + 
  PPDA_B + Challenges_B + ShareTklAttThird_B + ChallengesWonRate_B + TklWonRate_B + 
  AerialWonRate_B + LooseRecovWonRate_B + Directness_A + LongThroughSwitch_TotalPasses_A +
  ShareTouchAttThird_A


# USING K=5 CV ON TRAINING BEFORE TESTING
# Train LDA model using cross-validation
lda_model <- train(formula2, data = data_train, method = "lda", trControl = ctrl)
# View the cross-validation results
print(lda_model)
# Train LDA model on the entire training dataset
final_lda_model <- lda(formula2, data = data_train)
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
ggplot(data = coefficients, aes(x = reorder(row_names, Coefficients), y = Coefficients,fill=Coefficients)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_gradient(low="lightblue", high="darkred") + 
  labs(title="Best Predictors AWAY TEAM", y="", x="") +
  theme(panel.background = element_rect(color = "black", fill = "white"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  coord_flip()












#Kmeans - trials
set.seed(123)

selected_features <- c('TouchesAttThird_A', 'npxG_A', 'Shots_A', 'TeamA') 

team_Data_1 <- dplyr::select(MLSML, all_of(selected_features)) %>%
na.omit()

team_data_averages_1 <- team_Data_1 %>%
  group_by(TeamA) %>%
  summarise(across(c(TouchesAttThird_A, npxG_A, Shots_A), mean, na.rm = TRUE))

team_data_averages_1_scaled <- scale(select(team_data_averages_1, -TeamA))

k <- 3

set.seed(123)
kmeans_result <- kmeans(team_data_averages_1_scaled, centers = k)

team_data_averages_1$cluster <- kmeans_result$cluster

print(team_data_averages_1)



# 2d plot

team_data_averages_1$cluster <- as.factor(kmeans_result$cluster) 

ggplot(team_data_averages_1, aes(x = npxG_A, y = TouchesAttThird_A, color = cluster)) +
  geom_point(alpha = 0.6, size = 3) +  
  labs(title = "Cluster visualization with Feature1 and Feature2",
       x = "npxG_A",
       y = "TouchesAttThird_A") +
  scale_color_brewer(palette = "Set1") + 
  theme_minimal()  

# 3d plot
library(plotly)


cluster_plot <- plot_ly(data = team_data_averages_1, x = ~npxG_A, y = ~TouchesAttThird_A, z = ~Shots_A, type = "scatter3d", mode = "markers",
                marker = list(color = as.numeric(team_data_averages_1$cluster), colorscale = 'Viridis'))

cluster_plot









# find the best k
library(cluster)



selected_features_2 <- c(top_variables)
team_data_2 <- select(MLSML, all_of(top_variables)) %>% na.omit()


max_k <- 15  
wss <- numeric(max_k)
for (k in 1:max_k) {
  set.seed(123)  
  kmeans_result <- kmeans(team_data_2, centers=k, nstart=20)
  wss[k] <- sum(kmeans_result$withinss)
}

plot(1:max_k, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares", main="Elbow Method for Determining Optimal k", ylim=c(0, max(wss)))


#Kmeans2
set.seed(123)

selected_features_2 <- c(top_variables) 

team_data_2 <- dplyr::select(MLSML, TeamA,all_of(selected_features_2)) %>%
  na.omit()

team_data_averages_2 <- team_data_2 %>%
  group_by(TeamA) %>%
  summarise(across(c(top_variables), mean, na.rm = TRUE))

team_data_averages_2_scaled <- scale(select(team_data_averages_2, -TeamA))

k <- 3

set.seed(123)
kmeans_result_2 <- kmeans(team_data_averages_2_scaled, centers = k)

team_data_averages_2$cluster <- kmeans_result_2$cluster

print(team_data_averages_2)






# cluster coding

# Create a mapping of teams to their clusters from data_csv
cluster_mapping <- team_data_averages_2 %>%
  select(TeamA, cluster) %>%
  distinct()

# Rename columns for clarity and merging
colnames(cluster_mapping) <- c("Team", "Cluster")

# Merging
MLSML <- MLSML %>%
  left_join(cluster_mapping, by = c("TeamA" = "Team")) %>%
  rename(ClusterA = Cluster) %>%
  left_join(cluster_mapping, by = c("TeamB" = "Team")) %>%
  rename(ClusterB = Cluster)

# one-hot encoding
clusters <- c(1, 2, 3)
MLSML$ClusterA <- factor(MLSML$ClusterA, levels = clusters)
MLSML$ClusterB <- factor(MLSML$ClusterB, levels = clusters)

clusterA_encoded <- model.matrix(~ ClusterA - 1, data = MLSML)
clusterB_encoded <- model.matrix(~ ClusterB - 1, data = MLSML)

MLSML_coded <- cbind(MLSML, clusterA_encoded, clusterB_encoded)

print(head(MLSML_coded))


cluster_counts <- MLSML_coded %>%
  count(ClusterA, ClusterB) %>%
  mutate(Frequency = n)

# plot
ggplot(cluster_counts, aes(x = interaction(ClusterA, ClusterB), y = Frequency, fill = interaction(ClusterA, ClusterB))) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(x = "Cluster Combinations", y = "Count", title = "Numbers of clustering games") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Adjust the text angle for better readability



# Data_0 and groups selection
Data_0 <- MLSML_coded %>%
  select(Result_A,top_variables,
         ClusterA,ClusterB)

Data_0$Result_A <- ifelse(Data_0$Result_A == "L", 0, 1)

Data_1_1 <- Data_0 %>%
  filter(ClusterA == 1, ClusterB == 1) %>%
  select(-ClusterA, -ClusterB)

Data_1_2 <- Data_0 %>%
  filter(ClusterA == 1, ClusterB == 2) %>%
  select(-ClusterA, -ClusterB)

Data_1_3 <- Data_0 %>%
  filter(ClusterA == 1, ClusterB == 3) %>%
  select(-ClusterA, -ClusterB)

Data_2_1 <- Data_0 %>%
  filter(ClusterA == 2, ClusterB == 1) %>%
  select(-ClusterA, -ClusterB)

Data_2_2 <- Data_0 %>%
  filter(ClusterA == 2, ClusterB == 2) %>%
  select(-ClusterA, -ClusterB)

Data_2_3 <- Data_0 %>%
  filter(ClusterA == 2, ClusterB == 3) %>%
  select(-ClusterA, -ClusterB)

Data_3_1 <- Data_0 %>%
  filter(ClusterA == 3, ClusterB == 1) %>%
  select(-ClusterA, -ClusterB)

Data_3_2 <- Data_0 %>%
  filter(ClusterA == 3, ClusterB == 2) %>%
  select(-ClusterA, -ClusterB)

Data_3_3 <- Data_0 %>%
  filter(ClusterA == 3, ClusterB == 3) %>%
  select(-ClusterA, -ClusterB)




# linear programming
library(CVXR)


# 1_2 analysis
# linear programming
model_1_2 <- lm(Result_A ~ . , data = Data_1_2)

summary(model_1_2)

predictors <- names(coefficients(model_1_2)[-1])

variables <- setNames(lapply(predictors, function(x) Variable(1, name = x)), predictors)

coeffs <- coefficients(model_1_2)[predictors]  
contributions <- Map(function(coef, var) coef * var, coeffs, variables)

total_contribution <- Reduce(`+`, contributions)

objective <- Minimize(- (coefficients(model_1_2)["(Intercept)"] + total_contribution))

# feasible sets

constraints <- list()

numerical_columns <- names(Data_1_2)[sapply(Data_1_2, is.numeric) & names(Data_1_2) != "Result_A"]



# Function to sort points in clockwise order
order_points <- function(points) {
  # Calculate centroid
  centroid <- colMeans(points)
  
  # Calculate angles from centroid
  angles <- atan2(points[,2] - centroid[2], points[,1] - centroid[1])
  
  # Sort points by angle
  sorted_indices <- order(angles)
  sorted_points <- points[sorted_indices, ]
  
  return(sorted_points)
}


# 计算所有维度两两组合的凸包并存储凸包点
for (i in 1:(length(numerical_columns) - 1)) {
  for (j in (i + 1):length(numerical_columns)) {
    coords <- Data_1_2 %>% select(all_of(c(numerical_columns[i], numerical_columns[j])))
    
    hull_indices <- convhulln(as.matrix(coords))
    
    hull_points <- coords[hull_indices, , drop = FALSE]
    
    hull_points <- unique(hull_points)
    
    hull_points <- order_points(as.matrix(hull_points))
    
    # plot hull points
    plot <- ggplot(coords, aes_string(x = names(coords)[1], y = names(coords)[2])) +
      geom_point() +
      geom_polygon(data = as.data.frame(hull_points), aes_string(x = names(coords)[1], y = names(coords)[2]), 
                   color = "red", fill = NA) +
      ggtitle(paste("Convex Hull of", numerical_columns[i], "and", numerical_columns[j]))
    
    print(plot)
    
    # add constrains
    for (k in 1:(nrow(hull_points) - 1)) {
      # Compute the normal vector to the edge from point k to point k+1
      a <- hull_points[k+1, 2] - hull_points[k, 2]  # y2 - y1
      b <- hull_points[k, 1] - hull_points[k+1, 1]  # x1 - x2
      c <- -a * hull_points[k, 1] - b * hull_points[k, 2]  # -ax1 - by1
      # Add the constraint that all points must be on the left side of the edge
      constraints <- c(constraints, a * variables[[numerical_columns[i]]] + b * variables[[numerical_columns[j]]] <= -c)
    }
  }
}




problem <- Problem(objective,constraints)

result <- solve(problem)

optimized_values <- sapply(variables, function(v) result$getValue(v))

print(optimized_values)

optimal_value <- result$value

print(-optimal_value)




































