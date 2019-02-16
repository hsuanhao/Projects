# Predicting sentiment from product reviews

I used data from product review on Amazon.com to predict whether the sentiments about a product are positive or negative by implementing **logistic regression with L2 regularization** and **feature engineering**. 

In this project, I implemented two different feature vectorizations: **bag of words** and **TFIDF**. In the end, I calculated the area under the receiver operating characteristic curve (ROC AUC) on test dataset for these two models summarized below:

|       | bag of words |  TFIDF   |
|-------|--------------|----------|
|  AUC  | 0.9498       | 0.9642   |
