# Regression Model Selector

- By Joe Read and Ashray Shetty

Regression is a statistical technique that helps in quantifying the relationship between a taget variable and one or several features. The first step involves estimating the coefficient of the independent variable and then measuring the reliability of the estimated coefficient. This requires formulating a hypothesis, and based on the hypothesis, we can create a function.

If a manager wants to determine the relationship between the firm’s advertisement expenditures and its sales revenue, he will undergo the test of hypothesis. Assuming that higher advertising expenditures lead to higher sale for a firm. The manager collects data on advertising expenditure and on sales revenue in a specific period of time. This hypothesis can be translated into the mathematical function, where it leads to −

Y = A + Bx
Where Y is sales, x is the advertisement expenditure, A and B are constants.

After translating the hypothesis into this function, we can find the relationship between the dependent and independent variables. The idea is that if we know the value of our independent variable and our constants we can find calculate our expected value of depndent variable. This gives us a good way to predict and so regression analysis has become a commonplace in most predictive modelling. 

### There are however several types of regression:

**Simple regression** − One independent variable

**Multiple regression** − Several independent variables

**Polynomial Regression** - Polynomial expression between independent variables

**Lasso Regression** - Regression using shrinkage

**Ridge Regression** - Analyzing multiple regression data that suffer from multicollinearity.

Generally in a datascience project it is usual practice to run different models and then select the best among them. For the purpose of this project we thought that is there a way to create a general model selector .py file which given a clean dataset and a set of continuous features, would run these different test and provide an output of the necessary evaluation metrics to help decide between selection of models. 

The comparison parameters that we selected were:
1. the root mean squared error
2. r squared and adjusted r squared for training set
3. r squared and adjusted r squared for test set
4. 5 fold cross validation. 

This allows multiple qualities to be taken into account when choosing the most effective predictive model.

**Purpose for our project is as follows**
* Create a generalised model selector function
* Test our function on 3 different dataset


# Diamonds Dataset

We choose Diamonds dataset to begin with our initial stage of prototyping our general model selector function. Diamonds is classic dataset that contains the prices and other attributes of almost 54,000 diamonds. It's a great dataset for beginners learning to work with data analysis and visualization. Before we begin exploring and tesing our function on this dataset, it would be beneficial to have a basic conception regarding the anatomy of a diamond. The datset contains features which are related to a diamonds anatomy.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/Diamond.jpg alt="Anatomy of a Diamond" width="450"/>

This is what the first 3 rows of the dataset looks like:
* It has 9 features

|carat |	cut	color|	clarity|	depth|	table|	price|	x|	y|	z|
|------|-------------|---------|---------|-------|-------|---|---|---|
|0.23|	Ideal|	E|	SI2|	61.5|	55.0|	326|	3.95|	3.98|	2.43|
|0.21|	Premium|	E|	SI1|	59.8|	61.0|	326|	3.89|	3.84|	2.31|
|0.23|	Good|	E|	VS1|	56.9|	65.0|	327|	4.05|	4.07|	2.31|

Our main variable that we would like to predict for the sake of this project was price so we looked at its distribution and we found that it looks something like follows:

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/price.png alt="Distribution of Price" width="450"/>

This dataset is a classic dataset and so we did not really spend a lot of time in cleaning the dataset. We simply created a heatmap to check how correlation between variables. This would keep us informed while we interpret the results from our model selector. 

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/diamond_corr.png alt="Correlation matrix" width="450"/>

Once we ran our model sector function, providing it with the dataframe and only the continuous features, we obtained a result as following. The function splits the 70% of dataset as a training set and performs a 5 fold cross validation on the training data. It then predicts the target variable for our test dataset and returns the r squared value associated with the test. We found that atleast for the diamonds dataset, using a multivariate lasso regression with second order polynomial returns the best evaluation metrics.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/Diamonds_all_tests.png alt="Final output for diamonds dataset" width="1500"/>

Once we protyped and ensured that the function works with Diamonds dataset we wanted to take it forward and do analysis of other more interesting datasets. We begin our analysis by first predicting life expectancy of a country given certain features.


# Predicting Life Expectancy

The dataset was was collected from "WHO" and the United Nations website by Deeksha Russell and Duan Wang and is now stored on [Kaggle](https://www.kaggle.com/kumarajarshi/life-expectancy-who "Kaggle"). It contains 2939 observations about different countries between the years 2000 and 2015.

| columnns                                                                                      |
|--------------------------|----------------------------------|---------------------------------|
| Country                  | HIV\AIDS                         | Measles                         |
| Year                     | Hepatitis B                      | BMI                             |
| Life expectancy          | Polio                            | Status                          |
| Adult mortality          | Diphtheria                       | Prevalence for malnutrition 5-19|
| Infant mortality         | GDP                              | Schooling                       |
| Alcohol consumpton       | Population                       | Total expenditre on health      |

Given the fact that our purpose with working with the dataset was to check if our model selector function can be generalised across different datasets, we were not keen to spend a lot of time tin cleaning our dataset. This was also due to time constraints for finishing the project. We therefore decided to simplly drop all null values in our dataset instead of figure out clever ways to deal with them. So after removing all the null values we were left with 1319 observations on which we ran our regression functions. 

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/Life_corr.png alt="correlation matrix" width="800"/>

There were many features that have multicollinearity among each other. After performing our eda we decided to settle down for 5 features in our analysis. This is what their distribution looks like.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/pair_plots.png alt="Pair Plots" width="800"/>

We first decided to run an ols model to see how well these features can be used to predict our target variable of life expectancy. We can see that the model is good on its own giving us a adjusted r squared of 0.761. We can also see the influence of the individual features by looking at their coefficients. Not suprisingly, the only thing that has a negative effect on Life expectancy is HIV and death ratio. Suprising the impact of GDP in predicting life expectancy is quite low and schooling or education status seems to be the most important factor in predicting life expectancy.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/Ols_results.png alt="Results for OLS Model" width="500"/>

We ran the dataset on our function to check if there is a possibility of obtaining a better model that can predict life expectancy. We obtainied the following results. Suprisingly the best model that trains well and test well is 4th order ridge regression model. It is also the one that gave us the least root mean squared error.


<img src= https://github.com/JoeSRead/Diamonds/blob/master/Images/life.png alt="Final Output" width="1500"/>



## Take Home Message

Our suggestions for countries looking to increase their life expectancy is to focus their resources mainly towards increasing HIV awareness. Additionally, we recommend increasing promoting education and to invest more towards decreasng malnutrition.
