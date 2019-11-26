from sklearn import linear_model, metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, LinearRegression

import numpy as np 
import pandas as pd 


# Creates a dataframe containing all evalutation metrics, set up first!
def eval_df():
    
    evaluation = pd.DataFrame({'Model': [],
                               'Power': [],
                               'Hyper Parameters': [],
                               'Root Mean Squared Error (RMSE)':[],
                               'R-squared (training)':[],
                               'Adjusted R-squared (training)':[],
                               'R-squared (test)':[],
                               'Adjusted R-squared (test)':[],
                               '5-Fold Cross Validation':[]}
                              )
    return evaluation


# Adjusted r squared function
def adjustedR2(r2,n,p):
    return 1 - (1-r2)*((n-1)/(n-p -1))


# Splits dataframe on features/targests as well as a test/train split
def train_test(data, features, target, testsize = 0.25):
    
    X = data.loc[:, features]
    y = data.loc[:, target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testsize, random_state= 42)
    
    return X_train, X_test, y_train, y_test
    

    
    
def regression(data, features, target, evaluation_df, regtype = '', testsize = 0.25, poly_deg = 1):

    if regtype not in ['Multivariate', 'Lasso', 'Ridge']:
        print('Please choose a regtype from [Multivariate, Lasso, Ridge]')
    
    X_train, X_test, y_train, y_test= train_test(data, features, target, testsize=testsize)    
    
    hyp = 'None'
    
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    
    if  poly_deg > 1:
        poly = PolynomialFeatures(degree = poly_deg)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        order = 'Polynomial'
    else:
        order = 'Linear'
              
    if regtype == 'Multivariate':
        model = linear_model.LinearRegression()
              
    if regtype == 'Lasso':
        itr = int(input('How many interations?'))
        model = LassoCV(max_iter = itr, cv = 5)
              
    if regtype == 'Ridge':
        alph_lst = input('Enter alpha values seperated by spaces')
        split = alph_lst.split()
        alphs = []
        for x in split:
            alphs.append(float(x))
        model = RidgeCV(alphas = alphs, cv = 5)
            
    model.fit(X_train, y_train)
              
    coefficients = model.coef_
    coeff = dict(zip(features, coefficients))

    pred = model.predict(X_test)
              
    root_mean_square_error = float(format(np.sqrt(metrics.mean_squared_error(y_test, pred)),'.3f'))
              
    r2_train = float(format(model.score(X_train, y_train),'.3f'))
    r2_adj_train = float(format(adjustedR2(model.score(X_train, y_train), X_train.shape[0], len(features)),'.3f'))
              
    r2_test = float(format(model.score(X_test, y_test),'.3f'))
    r2_adj_test = float(format(adjustedR2(model.score(X_test, y_test), X_test.shape[0], len(features)),'.3f'))

    crossvalidation = KFold(n_splits = 5, shuffle = True, random_state = 42)
    cv = float(format(cross_val_score(model, X_train, y_train, cv = crossvalidation).mean(),'.3f'))
    
    if regtype == 'Lasso':
        alph = model.alpha_
        hyp = 'Iterations: {}  and  Alpha: {}'.format(itr, alph)
    
    if regtype == 'Ridge':
        alph = model.alpha_
        hyp = 'Alpha: {}'.format(alph)

#     print('Intercept: {:.2f}'.format(model.intercept_))
#     print('Coefficient: {}'.format(model.coef_))
      
    r = evaluation_df.shape[0]
    evaluation_df.loc[r] = ['{} {} Regression'.format(regtype, order), poly_deg, hyp, root_mean_square_error, r2_train, r2_adj_train, r2_test, r2_adj_test, cv]
              
    return evaluation_df

    
    