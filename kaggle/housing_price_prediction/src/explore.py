'''
As of right now, we have only:
-gradient_boost
-ridge_regression

For each model, we select the best one 
due to the best average cross validation 
score with 10 folds. 

See README for future work.
'''

import os, argparse, util, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def replace_na_with_none(total_df):
    # Start by replacing n/a elements with 'None' as a categoryId as per spec
    total_df['Alley'] = total_df['Alley'].cat.add_categories(['None']).fillna('None')
    total_df['BsmtQual'] = total_df['BsmtQual'].cat.add_categories(['None']).fillna('None')
    total_df['BsmtCond'] = total_df['BsmtCond'].cat.add_categories(['None']).fillna('None')
    total_df['BsmtExposure'] = total_df['BsmtExposure'].cat.add_categories(['None']).fillna('None')
    total_df['BsmtFinType1'] = total_df['BsmtFinType1'].cat.add_categories(['None']).fillna('None')
    total_df['BsmtFinType2'] = total_df['BsmtFinType2'].cat.add_categories(['None']).fillna('None')
    total_df['FireplaceQu'] = total_df['FireplaceQu'].cat.add_categories(['None']).fillna('None')
    total_df['GarageType'] = total_df['GarageType'].cat.add_categories(['None']).fillna('None')
    total_df['GarageFinish'] = total_df['GarageFinish'].cat.add_categories(['None']).fillna('None')
    total_df['GarageQual'] = total_df['GarageQual'].cat.add_categories(['None']).fillna('None')
    total_df['GarageCond'] = total_df['GarageCond'].cat.add_categories(['None']).fillna('None')
    total_df['PoolQC'] = total_df['PoolQC'].cat.add_categories(['None']).fillna('None')
    total_df['Fence'] = total_df['Fence'].cat.add_categories(['None']).fillna('None')
    total_df['MiscFeature'] = total_df['MiscFeature'].cat.add_categories(['None']).fillna('None')

    return total_df

def cluster_regression(df):

    bsmt_str = 'BsmtFinSF2'
    clf = linear_model.LinearRegression()
    
    #find the weights of the points given the bsmtfinsf2
    sample_weights = [0 if (df['BsmtFinSF2'][i] == 0) else 1 for i in range(0, len(df))]
    for i in range(0, len(df)):
        print(f'{df[bsmt_str][i]}, {sample_weights[i]}')
        
    x_in = [[df[bsmt_str][i]] for i in range(0, len(df))]
    clf.fit(X=x_in, y=df['LogSalePrice'], sample_weight=sample_weights)
    print(f'output of fitting: {clf.coef_}')
    
    sns.regplot(x='BsmtFinSF2',
                y='LogSalePrice',
               
                data=df)
    plt.show()
    
    return

def filter_zeros(col):    
    '''
    Takes a col in list form and passes each element through
    the filter to return a list of weights associated with each 
    element.
    '''    
    return [0 if (col[i] == 0) else 1 for i in range(0, len(col))]

def filter_all_ones(col):
    '''
    return a list of equal weights
    '''
    return [1 for i in range(0, len(col))]

def filter_on_value(col, value):
    '''
    returns a list of sample weights filtered on 
    the value passed in.
    '''
    return [0 if (col[i] == value) else 1 for i in range(0,len(col))]

def get_clf_cat(df, y_col, weights=None):
    print(df)
    train_y = df[y_col]
    train_x = df.drop([y_col], axis=1)
    if weights == None:
        weights = [1 for i in range(0, len(df))]
    clf = linear_model.LinearRegression()
    clf.fit(X=train_x, y=train_y, sample_weight=weights)
    return clf


def get_clf(df, x_col, y_col, weights=None, reshape=False):
    train_x = df[x_col]
    if reshape:
        train_x = train_x.values.reshape(-1,1)

    train_y = df[y_col]
    if weights == None:
        weights = [1 for i in range(0, len(df))]
    clf = linear_model.LinearRegression()
    clf.fit(X=train_x, y=train_y, sample_weight=weights)
    return clf

def implement_notes(df):
    train_y = df['LogSalePrice']
    clfs = {} # maps col_name to classifier
    
    df['LogLotArea'] = df['LotArea'].apply(np.log)
    df.drop(['LotArea'], axis=1)
    fit_df = pd.concat([df['LogLotArea'], train_y], axis=1)
    # now we just need to get a regression for this data
    clfs['LogLotArea'] = get_clf(df=fit_df, x_col='LogLotArea', y_col='LogSalePrice', reshape=True)
    
    max_year = df['YearBuilt'].max()
    df['YearsOld'] = df['YearBuilt'].apply(lambda x: max_year - x)
    df['YearsOld'] = df['YearsOld'].astype('float32')
    df.drop(['YearBuilt'], axis=1)
    fit_df = pd.concat([df['YearsOld'], train_y], axis=1)
    clfs['YearsOld'] = get_clf(df=fit_df, x_col='YearsOld', y_col='LogSalePrice', reshape=True)
    
    max_year = df['YearRemodAdd'].max()
    df['YearsRemodOld'] = df['YearRemodAdd'].apply(lambda x: max_year - x)
    df.drop(['YearRemodAdd'], axis=1)
    fit_df = pd.concat([df['YearsRemodOld'], train_y], axis=1)
    clfs['YearsRemodOld'] = get_clf(df=fit_df, x_col='YearsRemodOld', y_col='LogSalePrice', reshape=True)
    

    fit_df = pd.concat([df['MasVnrArea'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if fit_df['MasVnrArea'][i] == 0 else 1 for i in range(0, len(fit_df))]
    clfs['MasVnrArea'] = get_clf(df=fit_df, x_col='MasVnrArea', y_col='LogSalePrice', weights=sample_weights, reshape=True)
    
    fit_df = pd.concat([df['BsmtFinSF1'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if ((fit_df['BsmtFinSF1'][i] == 0) or (fit_df['BsmtFinSF1'][i] > 2000)) else 1 for i in range(0, len(fit_df))]
    clfs['BsmtFinSF1'] = get_clf(df=fit_df, x_col='BsmtFinSF1', y_col='LogSalePrice', weights=sample_weights, reshape=True)
    
    fit_df = pd.concat([df['BsmtFinSF2'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if fit_df['BsmtFinSF2'][i] == 0 else 1 for i in range(0, len(fit_df))]
    clfs['BsmtFinSF2'] = get_clf(df=fit_df, x_col='BsmtFinSF2', y_col='LogSalePrice', weights=sample_weights, reshape=True)
    
    fit_df = pd.concat([df['BsmtUnfSF'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if fit_df['BsmtUnfSF'][i] == 0 else 1 for i in range(0, len(fit_df))]
    clfs['BsmtUnfSF'] = get_clf(df=fit_df, x_col='BsmtUnfSF', y_col='LogSalePrice', weights=sample_weights, reshape=True)
    
    fit_df = pd.concat([df['TotalBsmtSF'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if ((fit_df['TotalBsmtSF'][i] == 0) or (fit_df['TotalBsmtSF'][i] > 2000)) else 1 for i in range(0, len(fit_df))]
    clfs['TotalBsmtSF'] = get_clf(df=fit_df, x_col='TotalBsmtSF', y_col='LogSalePrice', weights=sample_weights, reshape=True)

    fit_df = pd.concat([df['2ndFlrSF'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if fit_df['2ndFlrSF'][i] == 0 else 1 for i in range(0, len(fit_df))]
    clfs['2ndFlrSF'] = get_clf(df=fit_df, x_col='2ndFlrSF', y_col='LogSalePrice', weights=sample_weights, reshape=True)

    fit_df = pd.concat([df['LowQualFinSF'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if fit_df['LowQualFinSF'][i] == 0 else 1 for i in range(0, len(fit_df))]
    clfs['LowQualFinSF'] = get_clf(df=fit_df, x_col='LowQualFinSF', y_col='LogSalePrice', weights=sample_weights, reshape=True)

    fit_df = pd.concat([df['GarageArea'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if ((fit_df['GarageArea'][i] == 0) or (fit_df['GarageArea'][i] > 1200)) else 1 for i in range(0, len(fit_df))]
    clfs['GarageArea'] = get_clf(df=fit_df, x_col='GarageArea', y_col='LogSalePrice', weights=sample_weights, reshape=True)
    
    fit_df = pd.concat([df['WoodDeckSF'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if fit_df['WoodDeckSF'][i] == 0 else 1 for i in range(0, len(fit_df))]
    clfs['WoodDeckSF'] = get_clf(df=fit_df, x_col='WoodDeckSF', y_col='LogSalePrice', weights=sample_weights, reshape=True)
        
    fit_df = pd.concat([df['OpenPorchSF'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if ((fit_df['OpenPorchSF'][i] == 0) or (fit_df['OpenPorchSF'][i] > 400)) else 1 for i in range(0, len(fit_df))]
    clfs['OpenPorchSF'] = get_clf(df=fit_df, x_col='OpenPorchSF', y_col='LogSalePrice', weights=sample_weights, reshape=True)

    fit_df = pd.concat([df['EnclosedPorch'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if ((fit_df['EnclosedPorch'][i] == 0) or (fit_df['EnclosedPorch'][i] > 400)) else 1 for i in range(0, len(fit_df))]
    clfs['EnclosedPorch'] = get_clf(df=fit_df, x_col='EnclosedPorch', y_col='LogSalePrice', weights=sample_weights, reshape=True)

    fit_df = pd.concat([df['ScreenPorch'], train_y], axis=1)
    fit_df = fit_df.dropna()
    fit_df = fit_df.reset_index(drop=True)
    sample_weights = [0 if ((fit_df['ScreenPorch'][i] == 0) or (fit_df['ScreenPorch'][i] > 400)) else 1 for i in range(0, len(fit_df))]
    clfs['ScreenPorch'] = get_clf(df=fit_df, x_col='ScreenPorch', y_col='LogSalePrice', weights=sample_weights, reshape=True)
    
    # easy drop everything
    cols_to_drop = ['3SsnPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'Street',
                    'Alley', 'LotShape', 'Utilities', 'LandSlope', 'Condition2',
                    'HouseStyle', 'RoofMatl', 'Exterior2nd', 'Heating', 'Electrical', 
                    'BsmtHalfBath', 'KitchenAbvGr', 'Functional', 'MiscFeature']
    df = df.drop(cols_to_drop, axis=1)
    values = {'LotFrontage': 0, 'MasVnrArea': 0}
    df = df.fillna(value=values)
    
    for col, reg in clfs.items():
        print(f'{col}: {reg.coef_}')
    
    return df, clfs

def explore_data(train_data, dtypes):
    '''
    Idea here is to find ways of pre-processing the 
    data in order to create a better model. For predicting 
    housing prices, we don't necessarily need or want to 
    predict the outliars that well. If we predict the 
    average well, then we should be able to predict most 
    of the test points well. 
    
    We only care about the training data and the test data 
    doesn't matter for now since we are seeing what information
    is relevant.  Once we know what is relevant, then we do the 
    same thing to the test data. Ideally we make one big pipeline
    where we manipulate both the train and test data at the 
    same time once we know what we want to do.  It could be 
    that we want to make the manipulations into functions so that
    we just call the function on the frame and then we copy the 
    list of functions we call on the training data to the test data.
    
    '''
    
    columns = list(train_data)
    # Need to retype the columns
    for col in dtypes:
        if col in columns:
            # print(f'Found {col} in col list')
            train_data[col] = train_data[col].astype(dtypes[col])

    train_data_length = len(train_data)
    train_data = replace_na_with_none(train_data) # as per the spec
    train_data['LogSalePrice'] = train_data['SalePrice'].apply(np.log)
    
    train_data, regs = implement_notes(train_data) # we now have our child models
    print(train_data.info())
    
    # Now that we have our child models, we want to recompute all the 
    # rows to be these new prediction values
    new_y_hat = {}
    for col, reg in regs.items():
        print(f'Working with col: {col}')
        y_hat = reg.predict(train_data[col].values.reshape(-1,1))
        new_y_hat[col] = y_hat
    new_y_hat['LogSalePrice'] = train_data['LogSalePrice']
    master_df = pd.DataFrame(new_y_hat)
    description = master_df.describe()
    for col in description.columns:
        print('-'*40)
        print(description[col])
    
    master_reg = linear_model.LinearRegression()
    train_y = master_df['LogSalePrice']
    train_x = master_df.drop('LogSalePrice', axis=1)
    master_reg.fit(train_x, train_y) # HERE IS OUR MASTER REGRESSION
    
    return regs, master_reg
    
    # We want to try and detect outliers automatically and mark them
    # so that we can plot and label the outliers vs the non-outliers
    
    # For numerical values, outlier probably means anything beyond 3 sigma
    # For categorical data it might mean points that are beyond 3 sigma within 
    # each of the categories of the column
    
    # First we attempt to detect outliers, then we plot them against the 
    # log_sale_price to see if we are picking out good points
    
    for col in columns:
        print('-'*40)
        print(f'Working with column {col} with dtype={train_data[col].dtype}:')
        if str(train_data[col].dtype) != 'category':
            
            sns.scatterplot(x=col,
                            y='LogSalePrice',
                            data=train_data)
#             plt.show()
        else:
            sns.catplot(x=col,
                        y='LogSalePrice',
                        kind='swarm',
                        data=train_data)
            plt.show()
#             print(f'Skipping category column ({col}) for now...')
            

    sale_price = data['SalePrice']
    sale_price_str = 'SalePrice'
    
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
    print('Starting Ridge Regression exploration:')
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

    
    
    