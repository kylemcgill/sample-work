'''
As of right now, we have only:
-gradient_boost
-ridge_regression

For each model, we select the best one 
due to the best average cross validation 
score with 10 folds. 

See README for future work.
'''

import os, argparse, util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def explore_data(data, dtypes):
    sale_price = data['SalePrice']
    sale_price_str = 'SalePrice'
    columns = list(data)
    sns.set(style='darkgrid')
    
    for col in columns:
        print(f'Plot for: {col}')
        sns.relplot(x=col, y=sale_price_str, hue='Utilities', style='Utilities', data=data)
        plt.show() # code will block here until window has closed

def gradient_boost(x, y):
    '''
    Gradient Boost creates many random estimators
    and each estimator gets trained on a different 
    set of training values.  The model is one where 
    the collection of estimators votes on the right 
    answer.
    
    # Plot: n_estimators_per_n_training_points vs cross_validation_score
    
    x: DataFrame which holds all the training x's
    y: DataFrame which holds all the training y's
    '''
    print('Starting Gradient Boost Regression exploration:')
    n_folds = 10
    fold_list = [i for i in range(1, 11)]
    print(f'Check fold list: {fold_list}')
    n_rows = len(x)
    loss_functions = ['ls', 'lad', 'huber', 'quantile']
    percentages = np.arange(0.01, 0.1, 0.01)
    estimator_list = [int(n_rows*i) for i in percentages] # should return a list of estimators 
    print(f'Check estimator list: {estimator_list}')
    depths = [1,2,3,4,5]
    parameters_to_test = []
    
    print('Creating clasifiers we want to test...')
    # create the parameters we want to sample
    for current_loss_function in loss_functions:
        for current_estimator in estimator_list:
            for current_depth in depths:    
                clf = GradientBoostingRegressor(n_estimators=current_estimator,
                                              learning_rate=0.1,
                                              max_depth=current_depth, 
                                              random_state=0, 
                                              loss=current_loss_function)
                parameters_to_test.append(clf)
    # END create parameters
    
    #now test the parameters
    print(f'Cross validating classifiers with {len(parameters_to_test)} of them...')
    classifier_scores = {}
    count = 0
    for clf in parameters_to_test:
        count += 1
        print(f'working with {count}/{len(parameters_to_test)}:  {clf.get_params}')
        classifier_scores[clf] = cross_val_score(clf, x, y, cv=n_folds)
#         if(count > 10): break
    
    # Create a list of ParamScore objs to then sort
    avg_classifier_scores = []
    for k, v in classifier_scores.items():
        current_param = util.ParamScore(k, v.mean(), v.std())
        avg_classifier_scores.append(current_param)
    
    
    # Now we want to plot the parameters with the cross validation scores.
    # What we want to plot is holding all other values constant and varying
    # only one variable at a time to see any trends in how well the Gradient-
    # BoostRegressor does on cross validation. We end up with a lot of graphs.
    # Just sort by mean and stdev of the models and plot the top and bottom 3
    print('Sorting classifiers based on cross validation scores...')
    avg_classifier_scores = sorted(avg_classifier_scores, key=lambda param_score: param_score.score)
    
    print('Bottom 3 scores:')
    for i in range(0,3):
        print(f'{i}: {avg_classifier_scores[i].score}')
        
    print('Top 3 scores:')
    for i in range(len(avg_classifier_scores)-3, len(avg_classifier_scores)):
        print(f'{i}: {avg_classifier_scores[i].score}')
    
    # Output submit file using best_params
    last_score_index = len(avg_classifier_scores)-1
    best_clf = avg_classifier_scores[last_score_index].classifier
    
    return best_clf
    
def ridge_regression(x, y):
    print('Starting Gradient Boost Regression exploration:')
    n_folds = 10
    fold_list = [i for i in range(1, 11)]
    alphas_to_test = np.arange(1,100,10)
    
    parameters_to_test = []
    
    print('Creating clasifiers we want to test...')
    # create the parameters we want to sample
    for ridge in alphas_to_test:
        clf = linear_model.Ridge(alpha=ridge)
        parameters_to_test.append(clf)
    # END create parameters
    
    #now test the parameters
    print(f'Cross validating classifiers with {len(parameters_to_test)} of them...')
    classifier_scores = {}
    count = 0
    for clf in parameters_to_test:
        count += 1
        print(f'working with {count}/{len(parameters_to_test)}:  {clf.get_params}')
        classifier_scores[clf] = cross_val_score(clf, x, y, cv=n_folds)
#         if(count > 10): break
    
    # Create a list of ParamScore objs to then sort
    avg_classifier_scores = []
    for k, v in classifier_scores.items():
        current_param = util.ParamScore(k, v.mean(), v.std())
        avg_classifier_scores.append(current_param)
    
    
    # Just sort by mean and stdev of the models and plot the top and bottom 3
    print('Sorting classifiers based on cross validation scores...')
    avg_classifier_scores = sorted(avg_classifier_scores, key=lambda param_score: param_score.score)
    
    print('Bottom 3 scores:')
    for i in range(0,3):
        print(f'{i}: {avg_classifier_scores[i].score}')
        
    print('Top 3 scores:')
    for i in range(len(avg_classifier_scores)-3, len(avg_classifier_scores)):
        print(f'{i}: {avg_classifier_scores[i].score}')
    
    # Output submit file using best_params
    last_score_index = len(avg_classifier_scores)-1
    best_clf = avg_classifier_scores[last_score_index].classifier
    
    print(f'Best Classifier params: {avg_classifier_scores[last_score_index].classifier.get_params}')
    
    return best_clf

    
    
    