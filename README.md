# Regression Model Selector

- By Joe Read and Ashray Shetty

This repo contains the following files:
index.ipynb: Contains the jupyter notebook where the 2 datasets are run, along with their ouput
diamonds.csv and cleaned_life_expectancy.csv: Datasets used in the notebook
model.py: Contains raw code and functions for implemented in the notebook
presentation.pdf: slides explaining the purpose and outcome of the project
test_notebooks: is a folder containing jupyter notebooks used during prototyping.

Regression is a mathematical technique used in quantifying the relationship between a taget variable and one or more features. The first step involves estimating the impact each feature will have on the independent variable, and then tuning the coefficients to reliably predict the target variable. 

For instance if a manager wants to determine the relationship between the firm’s advertisement expenditures and its sales revenue, they will build a model linking the two, assuming, perhaps, that higher advertising expenditures lead to higher sale for a firm. The manager collects data on advertising expenditure and on sales revenue in a specific period of time which is then used to test the model which takes the form:

Y = A + Bx
Where Y is sales, x is the advertisement expenditure, A and B are constants to be determined

After translating the problem into this function, we can find the relationship between the dependent and independent variables. The idea is that if we know the value of our independent variable and our constants we can find calculate our expected value of depndent variable. This gives us a good way to predict and so regression analysis has become a commonplace in most predictive modelling. 

### There are however several types of regression:

**Simple regression** − One independent variable

**Multiple regression** − Several independent variables

**Polynomial Regression** - Polynomial expression between independent variables

**Lasso Regression** - Regression using shrinkage

**Ridge Regression** - Analyzing multiple regression data that suffer from multicollinearity.

Generally in a datascience project it is usual practice to run different models and then select the best among them. For the purpose of this project we tried to create a general model selector .py file which when given a clean dataset and a set of continuous features, would run a model and provide an output of the necessary evaluation metrics to help decide between selection of models. 

The comparison parameters that we selected were:
1. the root mean squared error
2. r squared and adjusted r squared for training set
3. r squared and adjusted r squared for test set
4. 5 fold cross validation. 

This allowed multiple qualities to be taken into account when choosing the most effective predictive model.

**Purpose for our project is as follows**
* Create a generalised model selector function
* Test our function on different dataset


# Diamonds Dataset

We chose the Diamonds dataset to begin prototyping our general model selector function. Diamonds is classic dataset that contains the prices and other attributes of almost 54,000 diamonds. It's a great dataset for beginners learning to work with data analysis and visualization. Before we begin exploring and tesing our function on this dataset, we needed to have a basic conception regarding the anatomy of a diamond as the datset contains many of their physical features.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/Diamond.jpg alt="Anatomy of a Diamond" width="450"/>

This is what the first 3 rows of the dataset looks like:
* It has 9 features

|carat |	cut	color|	clarity|	depth|	table|	price|	x|	y|	z|
|------|-------------|---------|---------|-------|-------|---|---|---|
|0.23|	Ideal|	E|	SI2|	61.5|	55.0|	326|	3.95|	3.98|	2.43|
|0.21|	Premium|	E|	SI1|	59.8|	61.0|	326|	3.89|	3.84|	2.31|
|0.23|	Good|	E|	VS1|	56.9|	65.0|	327|	4.05|	4.07|	2.31|

Our target variable that we would like to predict for this project was price, we looked at its logged distribution in order to roughly normalise it:

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/price.png alt="Distribution of Price" width="450"/>

This dataset is a classic dataset and so we did not really spend a lot of time in cleaning the dataset. We created a heatmap to check correlations between variables which we used to chose our set of predictor features.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/diamond_corr.png alt="Correlation matrix" width="450"/>

Our model sector function, once provided with the dataframe and only the continuous features, works as following: the function splits dataset into a training/test set and performs a 5 fold cross validation on the training data. It then predicts the target variable for our test dataset and returns the r squared value associated with the test. We found that for the diamonds dataset, using a multivariate lasso regression returns the best evaluation metrics.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/Diamonds_all_tests.png alt="Final output for diamonds dataset" width="1500"/>

Once we protyped and ensured that the function worked with the Diamonds dataset we wanted to take it further and produce analysis of other different datasets. We decided to use the model selector to find the best model for predicting life expectancy in a country given certain features.


# Predicting Life Expectancy

The dataset was was collected from "WHO" and the United Nations website by Deeksha Russell and Duan Wang and is now stored on [Kaggle](https://www.kaggle.com/kumarajarshi/life-expectancy-who "Kaggle"). It contains 2939 observations about different countries between the years 2000 and 2015.

| columnns                 |                                  |                                 |
|--------------------------|----------------------------------|---------------------------------|
| Country                  | HIV\AIDS                         | Measles                         |
| Year                     | Hepatitis B                      | BMI                             |
| Life expectancy          | Polio                            | Status                          |
| Adult mortality          | Diphtheria                       | Prevalence for malnutrition 5-19|
| Infant mortality         | GDP                              | Schooling                       |
| Alcohol consumpton       | Population                       | Total expenditre on health      |

Given that our aim with working with this dataset was to check if our model selector functiom generalises across different datasets, we were not invested in spending a lot of time time cleaning our dataset. We therefore decided to drop all null values in our dataset instead of trying to figure out ways of dealing with them. After removing all the null values we were left with 1319 observations on which we ran our regression functions. 

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/Life_corr.png alt="correlation matrix" width="800"/>

There were many features that have multicollinearity among each other. After performing our EDA we decided to settle for 5 features in our analysis. This is what their distribution looks like. We can see that none of them have any strong linear releationship or co-correlations with other features. The distribution of most features other than Schooling are not normal. We still decided to check how well these features can help predict life expectancy.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/pair_plots.png alt="Pair Plots" width="800"/>

We first decided to run an ols model to see how well these features can be used to predict our target variable of life expectancy. We can see that the model is good on its own giving us a adjusted r squared of 0.761. We can also see the influence of the individual features by looking at their coefficients. Not suprisingly, the only thing that has a negative effect on Life expectancy is HIV and death ratio. Suprising the impact of GDP in predicting life expectancy is quite low and schooling or education status seems to be the most important factor in predicting life expectancy. Although the OLS model is pretty good, it is not cross validated and so we dont know how well this model would predict our test set. Also we were interested to see if there are any other models that could improve the score of 0.761.

<img src=https://github.com/JoeSRead/Diamonds/blob/master/Images/Ols_results.png alt="Results for OLS Model" width="500"/>

We ran our model selector function on the dataset to see if there was a better model that could be used to predict life expectancy. We saw that the best model that describes both the training data well and the test data well is a 4th order Ridge regression model. This model was also the one that gave us the least root mean squared error. 


<img src= https://github.com/JoeSRead/Diamonds/blob/master/Images/life.png alt="Final Output" width="1500"/>



## Take Home Message

Our suggestions for countries looking to increase their life expectancy would be to focus their resources towards increasing HIV awareness. After that, we would recommend increasing education levels and investing more towards dealing with malnutrition.

## Future directions

The purpose of this project was to check if we can use a generalised approach to fiding the right model for a clean dataset. We have managed to show that it works for two very different datsets but there is still room for improvement. For instance in future it would better if we use AIC or BIC estimates which help us compare the between different models. 

We could also extend our project further by selecting or not selecting interactions/polynomial features depending on whether or not the AIC/BIC decreases when adding them in. This would be very useful in selecting actionable features to target.

Future work could also include ways to visualise different models performance to provide a more intuitive way of selecting between features and models.
