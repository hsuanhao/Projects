import numpy as np

def featureNormalize(X):
    """
    FEATURENORMALIZE: Normalizes the features in X
        featureNormalize(X) returns a normalized version of X where 
        the mean value of each feature is 0 and the standard deviation
        is 1. This is oftern a good preprocessing step to do 
        when working with learning algorithms.
    """
    
    X_norm = X         # X is m by 1 matrix, where m is number of samples
    mu = np.zeros((1,X.shape[1]))      # mean: Generate row vector
    sigma = np.zeros( (1, X.shape[1])) # standard deviation: Generate rwo vector
    
    mu = np.mean(X_norm,axis=0);         # Get mean value for each column
    sigma = np.std(X_norm,axis=0);       # Get standard deviation for each column
    X_norm = (X_norm - mu)/sigma
    return X_norm, mu, sigma

def effort(x):
    """
    Agent effort in hours is calculated by the following formula with given incentive:
    Y = 10*(1-exp(-x/400))
    Input:
       x: incentive
    Return:
       Y: effort
    """
    Y = 10*(1-np.exp(-x/400.0))
    return Y

def improvement(x):
    """
    % Improvement in renewal probability is calculated by the following formula with given agent effort
    Y = 20*(1-exp(-x/5))
    Input:
       x: agent effort in hours
    Return:
       Y: % improvement in renewal probability
    """
    Y = 20*(1-np.exp(-x/5.0))
    return Y

def error(y_pred, y_true):
    """
    Calculate least square error
    y_pred : prediction from the model
    y_true : true value from the data
    
    """
    m = len(y_true) # number of samples
    error = np.sum( (y_pred - y_true)**2 )/m
    return error