"""
Regression with general features in tensorflow framework
"""

# Author: Hsuan-Hao Fan

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression by using general features
    
        
    Attributes
    ----------
    
        
    Notes
    -----
    
    
    """

    def create_placeholders(self, n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.
    
        Parameters
        ----------
        n_x: scalar, int 
           number of features
        
        n_y: scalar, int
           number of classes
    
        Returns
        -------
        X: placeholder for the data input, of shape [n_x, None] and dtype "float"
        
        Y: placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
        Notes
        -----
        We use None because it let's us be flexible on the number of examples we will use for the placeholders.
        In fact, the number of examples during test/train is different.
        """
        X = tf.placeholder(tf.float64, [None, n_x])
        Y = tf.placeholder(tf.float64, [None, n_y])
    
        return X, Y    

    
    def initialize_parameters(self, X):
        """
        Initializes parameters to build linear model with tensorflow:
        
        Parameters
        ----------
        X: numpy array or sparse matrix of shape [m_samples,n_features]
           Training data
        
        Returns
        -------
        parameters: a dictionary of tensors containing W, b
        """
        m = X.shape[0] # number of samples
        n_features = X.shape[1] # number of features
        
        W = tf.get_variable("W", [n_features,1], dtype=tf.float64, initializer = tf.zeros_initializer())
        b = tf.get_variable("b", [1,1], dtype=tf.float64, initializer = tf.zeros_initializer())

        parameters = {"W": W, "b": b}
    
        return parameters
        
    def linear_model(self, X, parameters):
        """
        Implements the linear model: prediction = X * W + b
        
        Parameters
        ----------
        X: numpy array or sparse matrix of shape [m_samples,n_features]
           Training data
        
        parameters: a dictionary of tensors containing W of shape [n_features, 1], b of shape [m_samples, 1]
        """
        
        # Retrieve the parameters from the dictionary "parameters"
        W = parameters['W']
        b = parameters['b']
        
        prediction = tf.matmul(X,W) + b
        
        return prediction
        
        
        
        
    def compute_cost(self, prediction, Y):
        """
        Calculate cost function by using root mean square (RMS)
        
        Parameters
        ----------
        prediction: numpy array or sparse matrix of shape [m_samples,n_targets]
                    Training data
        
        Y : numpy array of shape [m_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary
            
        Returns
        -------
        cost: Tensor of the cost function
        """
        m = Y.shape[0] # number of samples
        
        rms = (prediction - Y)*(prediction - Y)
        cost = tf.reduce_mean(rms)
        
        return cost
    
    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
    
        Parameters
        ----------
        X: input data, of shape (number of examples, input size)
        Y: true "label" vector of shape (number of examples, 1)
        mini_batch_size: size of the mini-batches, integer
        seed: 
    
        Returns
        -------
        mini_batches: list of synchronous (mini_batch_X, mini_batch_Y)
        """
    
        m = X.shape[0]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)
    
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
        

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(m/mini_batch_size) 
        
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        return mini_batches
    
    
    def fit(self, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, 
            minibatch_size = 32, print_cost = True):
        """
        Implements a linear regression model by minimizing root mean square cost function.
    
        Parameters
        ----------
        X_train: training set, of shape [number of training examples, number of features ]
        Y_train: training set, of shape [number of test examples, output size]
        X_test: test set, of shape [number of test examples, number of features ]
        Y_test: test set, of shape [number of test examples, output size]
        learning_rate: learning rate of the optimization   
        num_epochs: number of epochs of the optimization loop
        minibatch_size: size of a minibatch
        print_cost: True to print the cost every 100 epochs
    
        Returns
        -------
        parameters -- parameters learnt by the model. They can then be used to predict.
        
        Notes
        -----
        Set minibatch_size = 1 to apply stochastic gradient descent (SGD)
        Set minibatch_size = number of training examples to apply batch gradient descent
        Set minibatch_size between 1 and number of training examples to apply mini batch gradient descent
        """
    
        tf.reset_default_graph()                      # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                         # to keep consistent results
        seed = 2                                      # to keep consistent results
        (m, n_x) = X_train.shape                      # (n_x: number of features, m : number of examples in the train set)
        n_y = Y_train.shape[1]                        # n_y : output size
        costs = []                                    # To keep track of the cost
    
        # Create Placeholders of shape (n_x, n_y)
        X, Y = self.create_placeholders(n_x, n_y)
    
        # Initialize parameters
        parameters = self.initialize_parameters(X_train)
    
    
        # Build the linear model in the tensorflow graph
        prediction = self.linear_model( X=X , parameters=parameters)
    
        # Cost function: Add cost function to tensorflow graph
        cost = self.compute_cost(prediction, Y)
    
        # Gradient Descent: Define the tensorflow optimizer. Use an GradientDescentOptimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost) 
    
    
        # Initialize all the variables
        init = tf.global_variables_initializer()
        
        # number of minibatches of size minibatch_size in the train set
        if m % minibatch_size == 0:
            num_minibatches = int(m / minibatch_size) 
        else:
            num_minibatches = int(m / minibatch_size) + 1
        
        #print(m, minibatch_size, num_minibatches)
        
        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:
        
            # Run the initialization
            sess.run(init)
        
            # Do the training loop
            for epoch in range(num_epochs):

                epoch_cost = 0.                       # Defines a cost related to an epoch
                seed = seed + 1
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    
                    epoch_cost += minibatch_cost
                
                    epoch_cost = epoch_cost/ num_minibatches
                

                # Print the cost every 100 epochs
                if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                    
                if print_cost == True and epoch % 10 == 0:
                    costs.append(epoch_cost)
                
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")
        
            
            if X_test is not None:
                #----- Need to correct in the following ------
                # Calculate the correct predictions
                correct_prediction = tf.equal(tf.argmax(prediction), tf.argmax(Y))
            

                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float64"))

                print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
                print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
    
    def predict(self, X, parameters):
        """
        Predicting Values by parameters obtained from model fitting
        
        Parameters
        ----------
        X: numpy array, input features
        parameters: dictionary, parameters obtained from model fitting
        
        Returns
        -------
        predictions: predict output values
        """
    
        W = tf.convert_to_tensor(parameters["W"])
        b = tf.convert_to_tensor(parameters["b"])
    
    
        params = {"W": W,
                  "b": b}
        m, n = X.shape
        
        x = tf.placeholder("float64", [m, n])
    
        prediction = self.linear_model( x , params) 
    
        with tf.Session() as sess:
            predictions = sess.run(prediction, feed_dict = {x: X})
        
        return predictions
    
class ExactLinearRegression:
    """
    Use closed form to fit linear regression model
    """
    
       
    def simple_linear_regression(self, input_feature, output):
        """
        Calculate parameters b and w of linear equation y= b + w*x by closed form.
    
        parameters:
        ----------
        input_feature: input feature for x
        output: output for y
    
        return:
        ------
        b : interception of linear equation
        w : slope of linear equation
        
        Notes
        -----
        In this simple linear regression model, we only consider single input feature.
        """
        mean_x = np.mean(input_feature)
        mean_y = np.mean(output)
        mean_xy = np.mean(input_feature*output)
        mean_xx = np.mean(input_feature*input_feature)

    
        w = (mean_x * mean_y - mean_xy)/(mean_x*mean_x - mean_xx)
        b = mean_y - w*mean_x
        
        parameters = {"W": w, "b": b}
        return parameters
    
    
    def predict(self, X, parameters):
        """
        Predicting Y Values by parameters obtained from model fitting, Y = W*X + b
        
        Parameters
        ----------
        X: numpy array, input feature
        parameters: dictionary, parameters obtained from model fitting
        
        Returns
        -------
        predictions: predict output values
        """
    
        W = parameters["W"]
        b = parameters["b"]
    
        predictions = X*W + b
    
        return predictions
    
    def inverse_regression_predictions(self, Y, parameters):
        """
        Predicting X Values by parameters obtained from model fitting, Y = W*X + b
        
        Parameters
        ----------
        Y: numpy array, output 
        parameters: dictionary, paramters obtained from model fitting
        
        Returns
        -------
        inverse_predictions: predict X values with given Y
        """
        W = parameters["W"]
        b = parameters["b"]
    
        inverse_predictions = (Y-b)/W
        
        return inverse_predictions
            
       
    def plot(self, X, Y, predictions, parameters, xlabel=None, ylabel=None):
        """
        Plot Y vs. X and fitting line
        
        Parameters
        ----------
        X: input feature
        Y: output values
        predictions: predicting values from fitting model
        parameters: parameters obtained from model fitting
        xlabel: optional, xlabel
        ylabel: optional, ylabel
        """
        W = parameters["W"]
        b = parameters["b"]
        y_eq = np.max(Y)*0.9  # y coordinate of equation
        x_eq = np.min(X)      # x coordinate of equation
        # Plot 
        plt.plot(X, Y, '.', X, predictions, '-')
        plt.text(x_eq, y_eq, "$y = %1.3f * x + (%1.3e ) $" % (W,b), fontsize=14, color="b")
        if (xlabel is not None) and (ylabel is not None): 
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        plt.show()
        
    def scores(self, Y, predictions):
        """
        Evaluate model by Residual Sum of Squares (RSS)
        
        Parameters
        ----------
        Y: output values
        predictions: predicting values from fitting model
        
        Returns:
        --------
        RSS: Residual Sum of Squares
        
        """
        
        
        RSS = np.sum( (Y - predictions)**2 )
        
        return RSS
        
        
        
