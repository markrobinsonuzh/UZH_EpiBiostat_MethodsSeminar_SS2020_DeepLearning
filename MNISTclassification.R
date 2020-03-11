# The problem we’re trying to solve here is to classify grayscale images of handwritten digits (28 × 28 pixels) 
# into their 10 categories (0 through 9). We’ll use the MNIST dataset, a classic in the machine-learning community, 
# which has been around almost as long as the field itself and has been intensively studied. 
# It’s a set of 60,000 training images, plus 10,000 test images, assembled by the National Institute of Standards and 
# Technology (the NIST in MNIST) in the 1980s. You can think of “solving” MNIST as the “hello world” of 
# deep learning—it’s what you do to verify that your algorithms are working as expected.

#install the Keras package
devtools::install_github("rstudio/keras")

# The Keras R interface uses the TensorFlow backend engine by default. 
# To install both the core Keras library as well as the TensorFlow backend use the
library(keras) # load the core Keras library
install_keras(tensorflow = "default") #the TensorFlow backend engine

#If you want to train your deep-learning models on a GPU,
#you can install the GPU-based version of the TensorFlow backend engine as follows
# install_keras(tensorflow = "gpu")

##############################################################################
################----------Preparing the Data------------#######################
##############################################################################
mnist <- dataset_mnist()
#x and y coordinates of the training and test datasets
# train_images and train_labels form the training set: the data from which the model will learn
train_images <- mnist$train$x
train_labels <- mnist$train$y
#The model will then be tested on the test set: test_images and test_labels
test_images <- mnist$test$x
test_labels <- mnist$test$y
#The images are encoded as 3D arrays, and the labels are a 1D array of digits, ranging from 0 to 9

#to check the structure of the arrays for the training data
str(train_images)
str(train_labels)

#to check the structure of the arrays for the test data
str(test_images)
str(test_labels)




# Next, we display the number of axes of the tensor train_images
length(dim(train_images))

#The shape
dim(train_images)

#this is its data type
typeof(train_images)

#For example, let's plot the fifth digit in this 3D tensor 
digit <- train_images[5,,]
plot(as.raster(digit, max = 255)) #The fifth sample in our dataset


#The x data is a 3-d array (images,width,height) of grayscale values . 
#To prepare the data for training we convert the 3-d arrays into matrices by
#reshaping width and height into a single dimension (28x28 images are flattened 
#into length 784 vectors). Then, we convert the grayscale values from integers 
#ranging between 0 to 255 into floating point values ranging between 0 and 1
# reshape
train_images <- array_reshape(train_images, c(nrow(train_images), 784))
dim(train_images)
test_images <- array_reshape(test_images, c(nrow(test_images), 784))
# rescale
train_images <- train_images / 255
dim(train_images)
test_images <- test_images / 255

# The y data is an integer vector with values ranging from 0 to 9. 
#To prepare this data for training we one-hot encode the vectors into 
#binary class matrices using the Keras to_categorical() function:
train_labels <- to_categorical(train_labels, 10)
test_labels <- to_categorical(test_labels, 10)

#We begin by creating a sequential model and then adding layers using the pipe (%>%) operator
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

#Next, compile the model with appropriate loss function, optimizer, and metrics:
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Use the fit() function to train the model for 30 epochs using batches of 128 images:
history <- model %>% fit(
  train_images, train_labels, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

# The history object returned by fit() includes loss and accuracy metrics which we can plot:
plot(history)

#Evaluate the model’s performance on the test data:
model %>% evaluate(test_images, test_labels)

#Generate predictions on new data:
model %>% predict_classes(test_images)

