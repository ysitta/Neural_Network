Congratulations! This is the end of Neural Network and our Machine Learning Specializations. The last part of this course is closed by filling this quiz.

To complete this assignment, you need to build your classification model to classify the categories of fashion image using Neural Network algorithms in one of the frameworks that is `Keras` by following these steps:

# 1 Data Preparation

Let us start our neural network experience by preparing the data first.  In this quiz, you will use the `fashionmnist` dataset. The data is stored as a csv file in this repository as fashionmnist folder. Please load the `fashionmnist` data under the `data_input` folder. The `fashionmnist` folder contains train and test set of 10 different categories for 28 x 28 pixel sized fashion images, use the following glossary for your target labels:

```
categories <- c("T-shirt", "Trouser", "Pullover", "Dress", 
    "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot")
```

## 1.1 Load the data

In this phase, please load and investigate our fashionmnist data and store it under `fashion_train` and `fashion_test` object. Please use the `read_csv()` function from the `readr` package to speed up when reading the data.

```
# your code here
```

Peek a `fashion_train` data by using `head()` function

```
# your code here
```

The `fashion_train` data consists of 60000 observations and 785 variables (1 target and 784 predictors). The predictors themselves contain the pixel of the image.

## 1.2 Convert the data to matrix

The data we have loaded above contains the value of pixels stored in **data frame**. Meanwhile, we have to convert the data into the matrix before we modeled the data, hence please convert the data to be matrix format using `data.matrix()` function and store it the `fashion_train` matrix as `train_m` and `fashion_test` matrix as `test_m`

```
# your code here
```

## 1.3 Cross Validation

After that, we should separate the predictors and target in our `train_m` and `test_m` data
```
# Predictor variables in `train_m`
train_x <-

# Predictor variables in `test_m`
test_x <-

# Target variables in `train_m`
train_y <-

# Target variables in `test_m`
test_y <-

```

## 1.4 Prepare training and testing set (change to an array)

Next, for the matrix variables that contain predictor variables, we should convert it to array shape. Please use the `array_reshape(data, dim(data))` to do that

```
train_x_array <- 
test_x_array <- 
```

## 1.5 Features scaling

The next preparation before the data is ready to be modeled feature scaling. Please answer this question first.

1. After the data reshape to an array, you must continue the preprocessed before training the network. If you inspect an arbitrary image in the training set, you will see that the pixel values fall in the range of 0 to 255. Then, what does the array divide by 255 do?
  - [ ] Convert the array between 0 to 255 into 0 to 1
  - [ ] Reshape the width and height into a single dimension since the data is a 3-d array (images, width, height)
  - [ ] Normalize the array between 0 to 255 into -1 to 1

Then scale the `train_x_array` and `test_x_array` by dividing to 255.

```
train_x.keras <- 
test_x.keras <- 
```

To prepare the data for the training model, we one-hot encode the vectors (`train_y`) into binary class matrices using `to_categorical()` function from `Keras` and stored it as `train_y.keras` object

```
# your code here
```

# 2 Build Neural Network Model

Before we applied the neural network to fashionmnist dataset, we should check the basic knowledge about the neural network by answering these questions below:

2. The below is the correct statements about Neural Networks, *EXCEPT*
  - [ ] Input layer is the first layer in Neural Network, the number of neurons depends on the predictor variable on the data
  - [ ] The initial weight  for each neuron is defined randomly
  - [ ] Activation functions are not doing any transformation to its previous layer
  - [ ] The neural network is called deep learning when it has more than one hidden layer

3. The neural network model is built to optimize (minimizing) the error, what kind of error in Regression case that we minimized?
  - [ ] Binary Crossentropy
  - [ ] Mean Absolute Error
  - [ ] Neuron weight

## 2.1 Build a model base using `keras_model_sequential()`

To organize the layers, we should create a base model, which is a Sequential model. Call a `keras_model_sequential()` function, and please pipe the base model with the model architecture.

## 2.2 Building Architecture (define layers, neurons, and activation function)

To define the architecture for each layer, we will build several models by tuning several parameters. Before building the architecture, we set the initializer to make sure the result will not change.

```
# your code here
```

First, create a model (stored it under `model_init`)by defining these parameters as:
- the first layer contains 32 nodes, relu activation function, 784 input shape
- the second layer contains 32 nodes, relu activation function
- the third layer contains 10 nodes, softmax activation function

Second, create a model (stored it under `model_bigger`)by defining these parameters as:
- the first layer contains 512 nodes, relu activation function, 784 input shape
- the second layer contains 512 nodes, relu activation function
- the third layer contains 10 nodes, softmax activation function

```
# your code here
```

4. In building the model architecture, we set several numbers of units. Below is the consideration using those number, *EXCEPT*
  - [ ] In the first layer, we use 784 input shape based on the number of our predictors
  - [ ] In the hidden and output layer, we use any even number
  - [ ] In the output layer, we use 10 that is the number of our categories

## 2.3 Building Architecture (define cost function and optimizer)

In this step, we still need to do several settings before the model is ready for training. Then, we should compile the model by defining the loss, optimizer type, and evaluation metrics. Please compile the model by setting these parameters:
- categorical crossentropy as loss function
- adam as the optimizer with learning rate 0.001
- used the accuracy as the metrics


```
# your code here
```

## 2.4 Fitting model in the training set (define epoch and batch size)

In this step, we fit our model using `epochs = 10` and `batch_size = 100` for those `model_init` and `model_bigger`. Please save the model in `history_init` and `history_bigger` object.

```
# your code here

```
5. In the fitting model above, we set `epochs = 10` means
  - [ ] The model does the feed-forward - back-propagation for all batch 10 times
  - [ ] The model does the weighting for all batch 10 times
  - [ ] The model divides the batch 10 times

```
# your code here
```

# 3 Predicting on the testing set

After we built our model, we then predict the testing (`test_x.keras`) data using the model that we have built. Please predict using `predict_classes()` function from `Keras` package and store it under `pred_init` and `pred_bigger`.

```
# your code here
```

# 4 Evaluating the neural network model

As the label is still in dbl type, then please decode the label based on its categories.

```
decode <- function(data){
  sapply(as.character(data), switch,
       "0" = "T-Shirt",
       "1" = "Trouser",
       "2" = "Pullover",
       "3" = "Dress",
       "4" = "Coat",
       "5" = "Sandal",
       "6" = "Shirt",
       "7" = "Sneaker",
       "8" = "Bag",
       "9" = "Boot")
}
```

Then, decode the `pred_init` and `pred_bigger` before we evaluate the model performance using a confusion matrix.

```
reference <- decode(test_y)
pred_decode_in <- 
pred_decode_big <- 
```

## 4.1 Confusion Matrix (classification)

After decoding the target variable, then you can evaluate the model using several metrics, in this quiz, please check the accuracy in the confusion matrix below.

Note: do not forget to do the explicit coercion `as.factor`.

```
# your code here
```

6. From the two confusion matrix, what can we infer?
  - [ ] The more the neuron, the model tends to overfit
  - [ ] The more the neuron, the model tends to underfit
  - [ ] The number of neuron in the hidden layer doesn't relate with underfit or overfit
  
## 4.2 Model Tuning

It turns out; our boss wants to get the best model, then he asks you to compare one model to another model (store it under `model_tuning`). Now, let us try to build the `model_tuning` by tuning these while compiling the model :
- used the sgd as the optimizer with learning rate 0.001
- the rest is the same with `model_init`

```
# your code here
```

7. The optimizer used to update the weight to minimize the loss function. What can you conclude from the model_init and model_tuning about the optimizer?
  - [ ] Optimizer Adam is more powerful than sgd
  - [ ] Optimizer Sgd is more powerful than adam
  - [ ] Both of the optimizers do not influence the model performance
  
8. From the two models above(`model_init`, and `model_tuning`), what is the best model for us to pick?
  - [ ] model_tuning because we have higher and balance accuracy between train and test
  - [ ] model_init because we have higher and balance accuracy between train and test
  - [ ] model_init because we have higher recall than model_tuning
 
