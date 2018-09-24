"""
FeatureProcessing is a module to process data or rescale data
"""

import numpy as np
   
def get_numpy_data(data_frame, features, output):
    """
    Convert Pandas DataFrame to Numpy array, and select input X and output Y
        
    Parameters:
    ----------
    data_frame: pandas DataFrame
        
    features: list of strings
          Selected features
              
    output: string
          Target
              
    Returns:
    --------
    X: numpy 2D array 
        (N, D) matrix, where N is number of samples and D is number of features
         
    Y: numpy 2D array
        (N, 1) matrix
    """
        
    # Convert dataframe into numpy array
    X = np.c_[data_frame[features]]
    Y = np.c_[data_frame[output]]
        
    return (X, Y)

def feature_normalize(X):
    """
    FEATURE_NORMALIZE: Normalizes the features in X
    
    Parameters:
    ----------
    X: numpyarray 
        
    Returns:
    --------
    X_norm: a normalized version of X 
    mu: the mean value of each feature 
    sigma: standard deviation of each feature
        
    Notes:
    ------
    This is often a good preprocessing step to do 
    when working with learning algorithms.
    """
    
    X_norm = X         # X is m by 1 matrix, where m is number of samples
    mu = np.zeros((1,X.shape[1]))      # mean: Generate row vector
    sigma = np.zeros( (1, X.shape[1])) # standard deviation: Generate rwo vector
    
    mu = np.mean(X_norm,axis=0);         # Get mean value for each column
    sigma = np.std(X_norm,axis=0);       # Get standard deviation for each column
    X_norm = (X_norm - mu)/sigma
        
    return X_norm, mu, sigma