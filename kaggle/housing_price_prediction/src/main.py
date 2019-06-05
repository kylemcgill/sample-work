import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import explore
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
parser.add_argument('-x', '--explore', help='Runs the explore package instead of building models',
                    action='store_true')

args = parser.parse_args()


housing_dtypes = {
    'Id'                :'int32',
    'MSSubClass'        :'category',
    'MSZoning'          :'category',
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
    'OverallQual'       :'int32',
    'OverallCond'       :'int32',
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
    'BsmtFullBath'      :'float32',
    'BsmtHalfBath'      :'float32',
    'FullBath'          :'float32',
    'HalfBath'          :'float32',
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

def condition_remap(x):
    if (x == 'Artery') or (x == 'Feedr'):
        return 'BIG_STREET'
    elif x == 'Norm':
        return 'NORMAL'
    elif (x == 'RRAn') or (x == 'RRNn') or (x == 'RRNe') or (x == 'RRAe'):
        return 'RAILROAD'
    elif (x == 'PosN') or (x == 'PosA'):
        return 'POSITIVE'
    else:
        return 'NONE'

def numerical_rating(x):
    if x == 'Po':
        return 1
    elif x == 'Fa':
        return 2
    elif x == 'TA':
        return 3
    elif x == 'Gd':
        return 4
    elif x == 'Ex':
        return 5
    else:
        return 0
    
def functional_remap(x):
    if x == 'Typ':
        return 7
    elif x == 'Min1':
        return 6
    elif x == 'Min2':
        return 5
    elif x == 'Mod':
        return 4
    elif x == 'Maj1':
        return 3
    elif x == 'Maj2':
        return 2
    elif x == 'Sev':
        return 1
    elif x == 'Sal':
        return 0

def num_to_cat(df):
    '''
    For making catagories out of numbers.
    '''
    return df
    
def cat_to_num(df):
    '''
    For making numerical values out of categories
    '''
    
    df['ExterQual'] = df['ExterQual'].apply(numerical_rating)
    df['ExterQual'] = df['ExterQual'].astype(float)

    df['ExterCond'] = df['ExterCond'].apply(numerical_rating)
    df['ExterCond'] = df['ExterCond'].astype(float)

    df['BsmtQual'] = df['BsmtQual'].apply(numerical_rating)
    df['BsmtQual'] = df['BsmtQual'].astype(float)

    df['BsmtCond'] = df['BsmtCond'].apply(numerical_rating)
    df['BsmtCond'] = df['BsmtCond'].astype(float)

    df['HeatingQC'] = df['HeatingQC'].apply(numerical_rating)
    df['HeatingQC'] = df['HeatingQC'].astype(float)

    df['KitchenQual'] = df['KitchenQual'].apply(numerical_rating)
    df['KitchenQual'] = df['KitchenQual'].astype(float)

    df['FireplaceQu'] = df['FireplaceQu'].apply(numerical_rating)
    df['FireplaceQu'] = df['FireplaceQu'].astype(float)
    
    df['Functional'] = df['Functional'].apply(functional_remap)
    df['Functional'] = df['Functional'].astype(float)
        
    return df

def fill_missing_values(df):
    print('Number of na values for each column:')
    for col in df.columns:
        if col == 'LogSalePrice':
            continue
        
        n_na = df[col].isna().sum()
        if n_na > 0:
            if str(df[col].dtype) == 'category':
                print(f'\tColumn {col} has {n_na} na values',
                    f'. Average value is {df[col].describe().top}')
                df[col] = df[col].fillna(df[col].describe().top)
                n_na = df[col].isna().sum()
                print(f'\tAfter fillna() {col} has {n_na} na values.')
            else:
                print(f'\tColumn {col} has {n_na} na values',
                  f'. Average value is {df[col].mean()}')
                df[col] = df[col].fillna(df[col].median())
                n_na = df[col].isna().sum()
                print(f'\tAfter fillna() {col} has {n_na} na values.')
    return df

def add_and_drop_cols(df):
    print(df.columns)
    df['LogLotArea'] = df['LotArea'].apply(np.log)
    df = df.drop(['LotArea'], axis=1)

    max_year = df['YearBuilt'].max()+10
    df['YearsOld'] = df['YearBuilt'].apply(lambda x: max_year - x)
    df['YearsOld'] = df['YearsOld'].astype('float32')
    df = df.drop(['YearBuilt'], axis=1)
    
    df['LogYearsOld'] = df['YearsOld'].apply(np.log)
    
    max_year = df['YearRemodAdd'].max()+10
    df['YearsRemodOld'] = df['YearRemodAdd'].apply(lambda x: max_year - x)
    df = df.drop(['YearRemodAdd'], axis=1)
    df['LogYearsRemodOld'] = df['YearsRemodOld'].apply(np.log)
    
    df = df.assign(TotalFinishedBsmtSF=lambda df: df.BsmtFinSF1+df.BsmtFinSF2)
    df = df.drop(['BsmtFinSF1', 'BsmtFinSF2'], axis=1)
    
    print(f'ex1={len(df.Exterior1st)} , ex2={len(df.Exterior2nd)}')
    column = [0 if str(df['Exterior1st'][i]) == str(df['Exterior2nd'][i]) else 1 for i in range(0,len(df))]
    df['HasTwoExteriors'] = column
    
    max_year = df['GarageYrBlt'].max()+10
    df['YearsOldGarage'] = df['GarageYrBlt'].apply(lambda x: max_year - x)
    df = df.drop(['GarageYrBlt'], axis=1)
    
    df['IsCulD'] = df['LotConfig'].apply(lambda x: 1 if x == 'CulDSac' else 0)
    df = df.drop(['LotConfig'], axis=1)
    
    df['NewCond1'] = df['Condition1'].apply(condition_remap)
    df = df.drop(['Condition1'], axis=1)
    
    df['BsmtBathTot'] = df.BsmtFullBath + df.BsmtHalfBath*0.5
    df = df.drop(['BsmtFullBath', 'BsmtHalfBath'], axis=1)
    
    df['BathTot'] = df.FullBath + df.HalfBath*0.5
    df = df.drop(['FullBath', 'HalfBath'], axis=1)

    # easy drop everything
    cols_to_drop = ['3SsnPorch', 'PoolArea', 'PoolQC', 'MiscVal', 'MoSold', 'YrSold', 'Street',
                    'Alley', 'LotShape', 'Utilities', 'LandSlope', 'Condition2',
                    'HouseStyle', 'RoofMatl', 'Exterior2nd', 'Heating', 'Electrical', 
                    'KitchenAbvGr', 'MiscFeature']
    df = df.drop(cols_to_drop, axis=1)
    values = {'LotFrontage': 0, 'MasVnrArea': 0}
    df = df.fillna(value=values)

    return df

def take_out_outliers(df):
    df = df[df['LotFrontage'] < 200]
    df = df[df['MasVnrArea'] < 1200]
    df = df[df['TotalBsmtSF'] < 3000]
    df = df[df['1stFlrSF'] < 3000]
    df = df[df['GrLivArea'] < 4000]
    df = df[df['GarageArea'] < 1200]
    df = df[df['OpenPorchSF'] < 400]
    df = df[df['TotalFinishedBsmtSF'] < 3000]
    
    df['OverallQual'] = df['OverallQual'].apply(lambda x: 3 if x <= 3 else x)
    df['BsmtQual'] = df['BsmtQual'].apply(lambda x: 2 if x < 2 else x)
    

    return df

# Try loading the data
train = pd.read_csv(args.indir + '/train.csv')
test = pd.read_csv(args.indir + '/test.csv')

if args.explore:
    explore.explore_data(train_data=train, dtypes=housing_dtypes)
    print('done.')
    quit()
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
    
    train['LogSalePrice'] = train['SalePrice'].apply(np.log)
    train = train.drop(['SalePrice'], axis=1)
    
    
    
    # 16% of rows don't have this value so drop it
#     train.drop('LotFrontage', axis=1, inplace=True)
#     test.drop('LotFrontage', axis=1, inplace=True)
    
    # Concatinate both sets so that we only have to use one lineto make manipulations
    total_df = pd.concat([train, test], sort=False)    

    # Need to retype the columns
    for col in housing_dtypes:
        if col in list(total_df.columns):
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
    
    total_df = fill_missing_values(total_df)
    total_df = add_and_drop_cols(total_df)
    total_df = cat_to_num(total_df)
    total_df = num_to_cat(total_df)
    total_df = pd.get_dummies(total_df)
    
    
    total_df.describe().to_csv(args.outdir + '/traindata.csv')
    print(total_df.describe())
    
    ntrain = len(train.index)
    ntest = len(test.index)
    train = total_df[:ntrain] # from 0 to end of train
    test = total_df[ntrain:] # from ntrain to end
    
    train = take_out_outliers(train)
    train.reset_index(drop=True)

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


train_y = train['LogSalePrice']
train_x = train.drop(['LogSalePrice'], axis=1)

test_y = test['LogSalePrice']
test_x = test.drop(['LogSalePrice'], axis=1)


best_clf = explore.ridge_regression(train_x, train_y)

best_clf.fit(train_x, train_y)

predictions = best_clf.predict(test_x)
out_pred = np.empty(len(predictions), dtype=np.float64)
np.exp(predictions, out=out_pred)

# The test data doesn't have any SalePrice on it so the only way we know 
# is by submitting a file for evaluation
with open(args.outdir + 'submit.csv', mode='w') as fout:
    fout.write('Id,SalePrice\n')
    for i in range(0, len(out_pred)):
        fout.write(f'{test_id.iloc[i]},{out_pred[i]}\n')

print('done.')