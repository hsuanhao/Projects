# Predicting the propensity to pay renewal premium and building an incentive plan for its agents to maximize the net revenue

This is **McKinsey Analytics Online Hackathon**. [[link](https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon-4/?utm_source=sendinblue&utm_campaign=Events_in_July_2018&utm_medium=email)] This even is hosted on AnalyticsVidhya in July 2018.

## Problem Statement

Your client is an Insurance company and they need your help in building a model to predict the propensity to pay renewal premium and build an incentive plan for its agents to maximise the net revenue (i.e. renewals - incentives given to collect the renewals) collected from the policies post their issuance.

You have information about past transactions from the policy holders along with their demographics. The client has provided aggregated historical transactional data like number of premiums delayed by 3/ 6/ 12 months across all the products, number of premiums paid, customer sourcing channel and customer demographics like age, monthly income and area type.

## Model Implemented

In this project, I tried **logistic regression** and **SVM** models with **stochastic gradient algorithm** by implementing [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) in scikit-learn; **Neural Network** with various hidden layers with **stochastic gradient algorithm** by applyting [MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) in scikit-learn. Also, I tried to include different features into the model training and experiment various alpha to avoid overfitting. Here, I only showed the best results I obtained by applying neural network with 1 hidden layer as shown in [insurance_renewal_NN_V10_best.ipynb](https://github.com/hsuanhao/Projects/blob/master/Predicting%20the%20propensity%20to%20pay%20renewal%20premium/insurance_renewal_NN_V10_best.ipynb).  
