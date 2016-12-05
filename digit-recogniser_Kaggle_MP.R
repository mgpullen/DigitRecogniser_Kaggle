# Clear workspace
rm(list = ls())

# Set seed
set.seed(123456)

# Import libraries
library(mxnet)
library(dplyr)

# Source other useful functions
source("randShuffle.R")

# Import the Kaggle training and testing data
train <- read.csv('data/train.csv', header = TRUE)
test  <- read.csv('data/test.csv',  header = TRUE)

# Generate noise matrix
noise <- data.frame(matrix(runif(42000 * 784, min = 50, max = 150), ncol = 784) / 100) # Same size as 'train'
train.noise.x <- train[, -1] * noise # get the X data and multiply it by the generated 'noise'
train.noise.y <- train[, 1] # isolate the y (prediction) column
train.noise <- cbind(train.noise.y, train.noise.x) # Combine y and X again
colnames(train.noise) <- colnames(train) # Make sure column names are consistent
train <- rbind(train, train.noise) # Row bind the new 'noisy' data to the original training data

# Randomise rows
train <- randShuffle(train)

# Extract X and y data
train.x <- train[, -1]
train.y <- train[, 1]

# Get 5% of the training data for validation/evaluation
valid.x <- t(train.x[80001:84000, ] / 255) # transpose and normalise the validation data between 0 - 1
train.x <- t(train.x[1:80000, ] / 255) # transpose and normalise the training data between 0 - 1
valid.y <- train.y[80001:84000]
train.y <- train.y[1:80000]
test.x  <- t(test / 255) # transpose and normalise the test data between 0 - 1

# Resize the dimensions of the X data sets
dim(valid.x) <- c(28, 28, 1, ncol(valid.x))
dim(train.x) <- c(28, 28, 1, ncol(train.x))
dim(test.x)  <- c(28, 28, 1, ncol(test.x))

# Uncomment if you want to look at a table of the values given for each image
# table(train.y)

# Identify the devices for MXNet to use
devices <- mx.cpu() # I don't have a GPU :(
mx.set.seed(0) # Set the MXNet seed

# Convolutional NN
data <- mx.symbol.Variable('data')

# First convolutional layer
conv1   <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20) # 20 filters of 5 x 5 size
act1    <- mx.symbol.Activation(data = conv1, act_type = "relu") # Linear rectifier activation
pool1   <- mx.symbol.Pooling(data = act1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2)) # Max pooling to decrease size

# Second convolutional layer
conv2   <- mx.symbol.Convolution(data = pool1, kernel = c(5, 5), num_filter = 50) # 50 filters of 5 x 5 size
act2    <- mx.symbol.Activation(data = conv2, act_type = "relu") # Linear rectifier activation
pool2   <- mx.symbol.Pooling(data = act2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2)) # Max pooling to decrease size

# First fully connected layer
flatten <- mx.symbol.Flatten(data = pool2) # Flatten data for fully connected layer
dropout <- mx.symbol.Dropout(flatten, p = 0.5) # Dropout layer to address over-fitting. Can change p to other values
fc1     <- mx.symbol.FullyConnected(data = dropout, num_hidden = 250) # First fully connected layer has 250 neurons. Number determined by trial and error at this stage
act3    <- mx.symbol.Activation(data = fc1, act_type = "relu") # Linear rectifier activation

# Second fully connected layer
fc2 <- mx.symbol.FullyConnected(data = act3, num_hidden = 10)  # Second fully connected layer has 10 neurons (one for each digit)

# Get output as a probability
softmax <- mx.symbol.SoftmaxOutput(data = fc2)

# Run the convolutional neural network model
model <- mx.model.FeedForward.create(softmax,
                                     X = train.x,
                                     y = train.y,
                                     ctx = devices,
                                     num.round = 50, # limited to 50 epochs due to computational constaints
                                     array.batch.size = 150, # batch size of 150 seems to be an okay...more testing required
                                     learning.rate = 0.1, # factor of ten either way from 0.1 degrades performance
                                     momentum = 0.8, # 0.8 seems to be the best value for the current data set
                                     eval.metric = mx.metric.accuracy,
                                     wd = 0.0001, # factor of ten either way from 0.0001 degrades performance
                                     eval.data=list(data = valid.x, label = valid.y),
                                     initializer = mx.init.uniform(0.1),
                                     # initializer = mx.init.Xavier(rnd_type = "gaussian", factor_type = "in", magnitude = 2.0),
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Calculate test predictions
preds.test <- predict(model, test.x)
pred.label.test <- max.col(t(preds.test)) - 1
# table(pred.label.test)

# Create submission.csv with the results of the test predictions
submission <- data.frame(ImageId = 1:dim(test.x)[4], Label = pred.label.test)
write.csv(submission, file = 'submission.csv', row.names = FALSE,  quote = FALSE)
