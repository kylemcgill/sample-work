import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor

print('Starting Housing Price Prediction...')

parser = argparse.ArgumentParser(description='Trains and tests models predicting house sell prices.')

# Add arguments here
parser.add_argument('-v', '--verbose', help='Prints more info to screen', action='store_true')
parser.add_argument('-d', '--debug', help='Saves preprocessed data and models to \'outdir\'', action='store_true')
parser.add_argument('-i', '--indir', help='Specifies the input dir to read both train and test data from',
                    action='store', default='../in/')
parser.add_argument('-o', '--outdir', help='Specifies the output dir to write any files to',
                    action='store', default='../out/')
parser.add_argument('-s', '--skip', help='Tells housing_price_prediction to skip the preprocessing step and go straight to modeling',
                    action='store_true')

args = parser.parse_args()


housing_dtypes = {
    'Id'                :'int32',
    'MSSubClass'        :'category',
    'MSZoning'         :'category',
    'LotFrontage'       :'float32',
    'LotArea'           :'int32',
    'Street'            :'category',
    'Alley'             :'category',
    'LotShape'          :'category',
    'LandContour'       :'category',
    'Utilities'         :'category',
    'LotConfig'         :'category',
    'LandSlope'         :'category',
    'Neighborhood'      :'category',
    'Condition1'        :'category',
    'Condition2'        :'category',
    'BldgType'          :'category',
    'HouseStyle'        :'category',
    'OverallQual'       :'category',
    'OverallCond'       :'category',
    'YearBuilt'         :'int32',
    'YearRemodAdd'      :'int32',
    'RoofStyle'         :'category',
    'RoofMatl'          :'category',
    'Exterior1st'       :'category',
    'Exterior2nd'       :'category',
    'MasVnrType'        :'category',
    'MasVnrArea'        :'float32',
    'ExterQual'         :'category',
    'ExterCond'         :'category',
    'Foundation'        :'category',
    'BsmtQual'          :'category',
    'BsmtCond'          :'category',
    'BsmtExposure'      :'category',
    'BsmtFinType1'      :'category',
    'BsmtFinSF1'        :'float32',
    'BsmtFinType2'      :'category',
    'BsmtFinSF2'        :'float32',
    'BsmtUnfSF'         :'float32',
    'TotalBsmtSF'       :'float32',
    'Heating'           :'category',
    'HeatingQC'         :'category',
    'CentralAir'        :'category',
    'Electrical'        :'category',
    '1stFlrSF'          :'float32',
    '2ndFlrSF'          :'float32',
    'LowQualFinSF'      :'float32',
    'GrLivArea'         :'float32',
    'BsmtFullBath'      :'category',
    'BsmtHalfBath'      :'category',
    'FullBath'          :'category',
    'HalfBath'          :'category',
    'BedroomAbvGr'      :'category',
    'KitchenAbvGr'      :'category',
    'KitchenQual'       :'category',
    'TotRmsAbvGrd'      :'category',
    'Functional'        :'category',
    'Fireplaces'        :'category',
    'FireplaceQu'       :'category',
    'GarageType'        :'category',
    'GarageYrBlt'       :'float32',
    'GarageFinish'      :'category',
    'GarageCars'        :'category',
    'GarageArea'        :'float32',
    'GarageQual'        :'category',
    'GarageCond'        :'category',
    'PavedDrive'        :'category',
    'WoodDeckSF'        :'float32',
    'OpenPorchSF'       :'float32',
    'EnclosedPorch'     :'float32',
    '3SsnPorch'         :'float32',
    'ScreenPorch'       :'float32',
    'PoolArea'          :'float32',
    'PoolQC'            :'category',
    'Fence'             :'category',
    'MiscFeature'       :'category',
    'MiscVal'           :'float32',
    'MoSold'            :'float32',
    'YrSold'            :'float32',
    'SaleType'          :'category',
    'SaleCondition'     :'category',
    'SalePrice'         :'float32'
}


# Try loading the data
train = pd.read_csv('../in/train.csv')
test = pd.read_csv('../in/test.csv')

train_id = train['Id']
test_id = test['Id']

print(f'Train shape: {train.shape}')
print(f'Train shape: {test.shape}')
# Preprocess the data
if not args.skip:
    print('Pre-processing raw data')
    
    # Drop the ids first 
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)

    # 16% of rows don't have this value so drop it
    train.drop('LotFrontage', axis=1, inplace=True)
    test.drop('LotFrontage', axis=1, inplace=True)
    
    # Concatinate both sets so that we only have to use one lineto make manipulations
    total_df = pd.concat([train, test], sort=False)
    col_list = total_df.columns.tolist()
    

    # Need to retype the columns
    for col in housing_dtypes:
        if col in col_list:
            # print(f'Found {col} in col list')
            total_df[col] = total_df[col].astype(housing_dtypes[col])

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

    # Now to check and see what the real amount of n/a responses are
    print('-'*60)
    print('Printing out missing elements as percentage of rows')
    n_rows = len(total_df.index)
    print(f'total number of rows: {n_rows}')
    modes = total_df.mode()
    print(modes)
    means = total_df.mean(skipna=True, numeric_only=True)
    print(means)

    for col in col_list:
        percent_missing = total_df[col].isna().sum() / n_rows

        if housing_dtypes[col] == 'category':
            # print(f'Filling category {col} with {modes[col][0]}')
            total_df[col] = total_df[col].fillna(modes[col][0])
            
        elif (col != 'SalePrice') & (housing_dtypes[col] != 'category'):
            # print(f'Filling numeric {col} with {means[col]}')
            total_df[col] = total_df[col].fillna(means[col])

        percent_missing_after = total_df[col].isna().sum() / n_rows
        # print(f'{col:<15} {percent_missing:>.2%} {percent_missing_after:>.2%}')
    
    total_df = pd.get_dummies(total_df)
    
    ntrain = len(train.index)
    ntest = len(test.index)
    train = total_df[:ntrain] # from 0 to end of train
    test = total_df[ntrain:] # from ntrain to end

    print(f'Train shape: {train.shape}')
    print(f'Train shape: {test.shape}')

    # Go back to working with the train data
    # Plot some thing to see what we are working with

    '''
    Because the sale price is on the interval [0, inf)
    we expect skew in the predictor variable. Plot this 
    to see our suspicion.  We want to predict a normal 
    variable 
    '''

# Now build the model after pre-processing
print('Building model')

train_y = train['SalePrice']
log_train_y = np.empty(len(train_y), dtype=float)
np.log(train_y, out=log_train_y)
train_x = train.drop(['SalePrice'], axis=1)

test_y = test['SalePrice']
test_x = test.drop(['SalePrice'], axis=1)

# Define the models here
clf_reg = linear_model.LinearRegression()
clf_ridge = linear_model.Ridge(alpha=.5)
clf_tree = tree.DecisionTreeRegressor()
clf_grad = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
        max_depth=1, random_state=0, loss='ls')

reg_scores = cross_val_score(clf_reg, train_x, log_train_y, cv=10)
print(f'Linear Regression scores: {reg_scores}')
print(f'mean: {reg_scores.mean()}, std: {reg_scores.std()}')

ridge_scores = cross_val_score(clf_ridge, train_x, log_train_y, cv=10)
print(f'Linear Ridge Regression scores: {ridge_scores}')
print(f'mean: {ridge_scores.mean()}, std: {ridge_scores.std()}')

tree_scores = cross_val_score(clf_tree, train_x, log_train_y, cv=10)
print(f'Tree Regresion scores: {tree_scores}')
print(f'mean: {tree_scores.mean()}, std: {tree_scores.std()}')

grad_scores = cross_val_score(clf_grad, train_x, log_train_y, cv=10)
print(f'Graidient Boost scores: {grad_scores}')
print(f'mean: {grad_scores.mean()}, std: {grad_scores.std()}')


clf_ridge.fit(train_x, log_train_y)

# Create our predictions on the test set
predictions = clf_ridge.predict(test_x)
out_pred = np.empty(len(predictions), dtype=np.float64)
np.exp(predictions, out=out_pred)

# The test data doesn't have any SalePrice on it so the only way we know 
# is by submitting a file for evaluation
with open(args.outdir + 'submit.csv', mode='w') as fout:
    fout.write('Id,SalePrice\n')
    for i in range(0, len(out_pred)):
        fout.write(f'{test_id.iloc[i]},{out_pred[i]}\n')



print('done.')