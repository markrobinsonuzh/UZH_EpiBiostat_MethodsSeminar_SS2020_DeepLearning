# Example tabular data

library(caret)
library(lime)

# Split up the data set
iris_test <- iris[1:5, 1:4]
iris_train <- iris[-(1:5), 1:4]
iris_lab <- iris[[5]][-(1:5)] # targets for training data

# Create Random Forest model on iris data
model <- train(iris_train, iris_lab, method = 'rf')

# Create an explainer object
explainer <- lime(iris_train, model,bin_continuous = TRUE,n_bins = 4) # if bin_continuous = TRUE categorize, then will split by quantiles

# Explain new observation
explanation <- explain(iris_test, explainer, n_labels = 1, n_features = 3) # n_features = complexity of the interpretable model

# n_labels= number of top class labels to explain a.k.a most likely classes for an specific observation

explanation

# And can be visualised directly
plot_features(explanation,cases = 5)


