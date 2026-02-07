import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p

df_train = pd.read_csv(r'F:\housing_prices\source\train.csv')
df_test  = pd.read_csv(r'F:\housing_prices\source\test.csv')

y_train = df_train[['Id', 'SalePrice']].copy()
df_train.drop('SalePrice', axis=1, inplace=True)

outliers_idx = df_train['GrLivArea'] > 4000
df_train = df_train[~outliers_idx]
y_train = y_train[~outliers_idx]

y_train['SalePrice'] = np.log1p(y_train['SalePrice'])

ntrain = df_train.shape[0]
ntest = df_test.shape[0]

all_data = pd.concat([df_train, df_test], axis=0, ignore_index=True)
all_ID = all_data['Id']
all_data.drop('Id', axis=1, inplace=True)

if 'Utilities' in all_data.columns:
    all_data.drop('Utilities', axis=1, inplace=True)

for col in all_data.columns:

    if col in (
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2', 'Fence',
        'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'MasVnrType',
        'MiscFeature', 'PoolQC'
    ):
        all_data[col] = all_data[col].fillna('None')

    elif col == 'LotFrontage':
        all_data[col] = all_data.groupby('Neighborhood')[col] \
                                 .transform(lambda x: x.fillna(x.median()))

    elif col in (
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
        'GarageArea', 'GarageCars', 'MasVnrArea'
    ):
        all_data[col] = all_data[col].fillna(0)

    elif col == 'GarageYrBlt':
        # Missing garages -> 0 + indicator
        all_data['GarageYrBlt_missing'] = all_data[col].isnull().astype(int)
        all_data[col] = all_data[col].fillna(0)

    elif col in (
        'MSZoning', 'Electrical', 'KitchenQual',
        'Exterior1st', 'Exterior2nd', 'SaleType'
    ):
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    elif col == 'Functional':
        all_data[col] = all_data[col].fillna('Typ')

    else:
        all_data[col] = all_data[col].fillna(0)

# ------------------------------
# Feature engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']
all_data['TotalSqFoot'] = all_data['TotalSF'] + all_data['TotalPorchSF']
all_data['TotalBath'] = (all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath'] +
                         all_data['FullBath'] + 0.5*all_data['HalfBath'])

all_data['Age'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBasement'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

# Interaction features
all_data['GrLivArea_by_TotalSF'] = all_data['GrLivArea'] / (all_data['TotalSF'] + 1)
all_data['OverallQual_by_Age'] = all_data['OverallQual'] / (all_data['Age'] + 1)

qual_map = {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
ordinal_cols = ('ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual',
                'BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC')

for col in ordinal_cols:
    all_data[col] = all_data[col].map(qual_map).fillna(0).astype(int)

for col in ('MSSubClass', 'YrSold', 'MoSold'):
    all_data[col] = all_data[col].astype(str)

numeric_feats = all_data.select_dtypes(exclude='object').columns
numeric_feats = [c for c in numeric_feats if c not in ordinal_cols]

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]

for feat in skewed_feats.index:
    all_data[feat] = boxcox1p(all_data[feat], 0.15)

all_data = pd.get_dummies(all_data)

X_train = all_data.iloc[:ntrain, :].copy()
X_test  = all_data.iloc[ntrain:, :].copy()

X_train = pd.concat([all_ID[:ntrain], X_train], axis=1)
X_test  = pd.concat([all_ID[ntrain:], X_test], axis=1)

X_train.to_csv(r'F:\housing_prices\export\train_cleaned.csv', index=False)
X_test.to_csv(r'F:\housing_prices\export\test_cleaned.csv', index=False)
y_train.to_csv(r'F:\housing_prices\export\y_train.csv', index=False)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Any NaNs?", X_train.isnull().any().any(), X_test.isnull().any().any())

# Check infinities for numeric columns only
numeric_train = X_train.select_dtypes(include=[np.number])
numeric_test  = X_test.select_dtypes(include=[np.number])
print("Any infinities?", np.isinf(numeric_train.values).any(), np.isinf(numeric_test.values).any())