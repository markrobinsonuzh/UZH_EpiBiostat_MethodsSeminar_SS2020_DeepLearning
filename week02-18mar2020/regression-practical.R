# Methods Seminar Deep Learning SS20
# Week 02: Scalar/vector regression and classification
# Judith, Deepak, Lucas

# This script covers the practical part on regression with continuous
# targets using keras in R
# It also contains code for k-fold cross validation

set.seed(2410)

# Depdendencies -----------------------------------------------------------

library(keras)


# Linear regression on simulated data -------------------------------------

# We will generate a dataset with a continuous target and a single predictor
# with known intercept and slope

n <- 100
intercept <- 2
slope <- 1
predictor <- runif(n, min = -10, max = 10)
noise <- rnorm(n, mean = 0, sd = 3)
target <- intercept + slope * predictor + noise

# plot(predictor, target, pch = 20)

# Our beloved linear model
m <- lm(target ~ predictor)
summary(m)

# Now we will reproduce the same results with a neural network with one neuron

network <- keras_model_sequential() %>% 
  layer_dense(units = 1, input_shape = 1)

network %>% 
  compile(optimizer = "rmsprop", loss = "mse")

network # 2 tunable parameters corresponding exactly to slope and intercept

network %>% 
  fit(predictor, target, epochs = 100, batch_size = 1)

coef(m) # we arrive at virtually the same results after training long enough
network$get_weights()


# Deep learning model for regression --------------------------------------

bh <- dataset_boston_housing()
# 404 training, 102 test
# Boston housing data
# response: median housing value

c(c(train_data, train_targets), c(test_data, test_targets)) %<-% bh

train_data <- scale(train_data)
test_data <- scale(test_data)

build_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
  
  return(model)
}

mod <- build_model()

mod %>% 
  fit(train_data, train_targets, epochs = 50, batch_size = 1)

mod %>% 
  evaluate(test_data, test_targets) # mae 2.6k dollars

# k-fold cv with more epochs and keeping track of performance on the fly

k <- 4
num_epochs <- 50
all_mae_hist <- NULL
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

for (i in 1:k) {
  cat("Processing fold #", i, "\n")
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  partial_train_d <- train_data[-val_indices, ]
  partial_train_t <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(partial_train_d, partial_train_t,
                           validation_data = list(val_data, val_targets),
                           epochs = num_epochs, batch_size = 1, verbose = 1)
  
  mae_history <- history$metrics$val_mae
  all_mae_hist <- rbind(all_mae_hist, mae_history)
}

all_mae_hist
(idx <- which.min(apply(all_mae_hist, 2, mean))) # lets take that for our final model

# fit final model
model <- build_model()

model %>% fit(train_data, train_targets, epochs = idx, batch_size = 8)

results <- model %>% 
  evaluate(test_data, test_targets)

results # 2.8k-ish dollars mae on the test set

