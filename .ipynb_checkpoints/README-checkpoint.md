# Diamonds

We have used the canonical Diamonds data set as a base to produce a table comparing various regression models. The table is produced by a function that when passed a clean data set and a desired model returns a row in the table describing the model along with several parameters for comparison. The idea here was to produce a tool for finding the best model for any given clean data set. The function can fit according to linear regression, multivariate linear regression, and polynomial regression (of any desired order), as well as use different cost functions such as ridge or lasso to find the best model.  


The comparison parameters that we selected were the root mean squared error, r squared and adjusted r squared (for both test and train data), and 5 fold cross validation. This allows multiple qualities to be taken into account when choosing the most effective predictive model.
