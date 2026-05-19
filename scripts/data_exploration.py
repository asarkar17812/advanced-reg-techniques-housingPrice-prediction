import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

SRC = r'F:\housing_prices\source'
OUT = r'F:\housing_prices\export'

df_train = pd.read_csv(rf'{SRC}\train.csv')
df_test  = pd.read_csv(rf'{SRC}\test.csv')

y_train = df_train[['Id', 'SalePrice']].copy()
df_train.drop('SalePrice', axis=1, inplace=True)

# Drop the two extreme GrLivArea outliers flagged by the dataset author
outliers_idx = df_train['GrLivArea'] > 4000
df_train = df_train[~outliers_idx]
y_train = y_train[~outliers_idx]

# Log-transform the target: SalePrice is right-skewed (skew ≈ 1.88);
# log1p makes residuals roughly normal and turns RMSE into RMSLE on the original scale.
y_train['SalePrice'] = np.log1p(y_train['SalePrice'])

ntrain = df_train.shape[0]
ntest = df_test.shape[0]

all_data = pd.concat([df_train, df_test], axis=0, ignore_index=True)
all_ID = all_data['Id']
all_data.drop('Id', axis=1, inplace=True)

# Utilities is all "AllPub" except one row → no signal
if 'Utilities' in all_data.columns:
    all_data.drop('Utilities', axis=1, inplace=True)

NONE_FEATS = {
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'Fence',
    'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'MasVnrType',
    'MiscFeature', 'PoolQC',
}
ZERO_FEATS = {
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
    'GarageArea', 'GarageCars', 'MasVnrArea',
}
MODE_FEATS = {
    'MSZoning', 'Electrical', 'KitchenQual',
    'Exterior1st', 'Exterior2nd', 'SaleType',
}

for col in all_data.columns:
    if col in NONE_FEATS:
        all_data[col] = all_data[col].fillna('None')
    elif col == 'LotFrontage':
        # neighborhoods have very different median frontages
        all_data[col] = all_data.groupby('Neighborhood')[col].transform(lambda x: x.fillna(x.median()))
    elif col in ZERO_FEATS:
        all_data[col] = all_data[col].fillna(0)
    elif col == 'GarageYrBlt':
        all_data['GarageYrBlt_missing'] = all_data[col].isnull().astype(int)
        all_data[col] = all_data[col].fillna(0)
    elif col in MODE_FEATS:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    elif col == 'Functional':
        all_data[col] = all_data[col].fillna('Typ')
    else:
        all_data[col] = all_data[col].fillna(0)

# Size aggregates
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalPorchSF'] = (
    all_data['OpenPorchSF'] + all_data['EnclosedPorch']
    + all_data['3SsnPorch'] + all_data['ScreenPorch']
    + all_data['WoodDeckSF']
)
all_data['TotalSqFoot'] = all_data['TotalSF'] + all_data['TotalPorchSF']
all_data['TotalBath'] = (
    all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
    + all_data['FullBath'] + 0.5 * all_data['HalfBath']
)

# Some test rows have YrSold < YearBuilt → clip negative ages to 0
all_data['Age']      = (all_data['YrSold'] - all_data['YearBuilt']).clip(lower=0)
all_data['RemodAge'] = (all_data['YrSold'] - all_data['YearRemodAdd']).clip(lower=0)
all_data['IsNew']         = (all_data['Age'] == 0).astype(int)
all_data['IsRemodeled']   = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)

all_data['HasGarage']     = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBasement']   = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace']  = (all_data['Fireplaces'] > 0).astype(int)
all_data['HasPool']       = (all_data['PoolArea'] > 0).astype(int)
all_data['Has2ndFloor']   = (all_data['2ndFlrSF'] > 0).astype(int)

# Ordinal mappings: these categoricals have natural orderings that get lost under one-hot.
qual_map          = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
bsmt_exposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
bsmt_fin_map      = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
garage_fin_map    = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
fence_map         = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
functional_map    = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
paved_map         = {'N': 0, 'P': 1, 'Y': 2}
slope_map         = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
shape_map         = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}

ordinal_specs = {
    'ExterQual': qual_map, 'ExterCond': qual_map, 'HeatingQC': qual_map,
    'KitchenQual': qual_map, 'BsmtQual': qual_map, 'BsmtCond': qual_map,
    'FireplaceQu': qual_map, 'GarageQual': qual_map, 'GarageCond': qual_map,
    'PoolQC': qual_map,
    'BsmtExposure': bsmt_exposure_map,
    'BsmtFinType1': bsmt_fin_map, 'BsmtFinType2': bsmt_fin_map,
    'GarageFinish': garage_fin_map,
    'Fence': fence_map,
    'Functional': functional_map,
    'PavedDrive': paved_map,
    'LandSlope': slope_map,
    'LotShape': shape_map,
}
for col, m in ordinal_specs.items():
    all_data[col] = all_data[col].map(m).fillna(0).astype(int)
ordinal_cols = list(ordinal_specs.keys())

# CentralAir is binary
all_data['CentralAir'] = (all_data['CentralAir'] == 'Y').astype(int)

# Quality × size interactions — repeatedly cited as the strongest single signals in Ames
all_data['OverallQual_x_TotalSF']   = all_data['OverallQual']  * all_data['TotalSF']
all_data['OverallQual_x_GrLivArea'] = all_data['OverallQual']  * all_data['GrLivArea']
all_data['Qual_x_Cond']             = all_data['OverallQual']  * all_data['OverallCond']
all_data['ExterQual_x_TotalSF']     = all_data['ExterQual']    * all_data['TotalSF']
all_data['KitchenQual_x_TotalSF']   = all_data['KitchenQual']  * all_data['TotalSF']
all_data['BsmtQual_x_TotalBsmtSF']  = all_data['BsmtQual']     * all_data['TotalBsmtSF']

all_data['GrLivArea_by_TotalSF']    = all_data['GrLivArea'] / (all_data['TotalSF'] + 1)
all_data['OverallQual_by_Age']      = all_data['OverallQual'] / (all_data['Age'] + 1)

# These are numeric but nominal (no order in MSSubClass; YrSold/MoSold are seasonality, not magnitude)
for col in ('MSSubClass', 'YrSold', 'MoSold'):
    all_data[col] = all_data[col].astype(str)

# Skewness correction — only on continuous numeric features.
# Exclude ordinal columns, binary indicators, and 1–5 grade scales that aren't continuous.
binary_cols = [c for c in all_data.columns
               if all_data[c].dtype != 'object' and all_data[c].dropna().nunique() <= 2]
exclude_skew = set(ordinal_cols) | set(binary_cols) | {'OverallQual', 'OverallCond'}
numeric_feats = [c for c in all_data.select_dtypes(exclude='object').columns if c not in exclude_skew]

skewed = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed = skewed[abs(skewed) > 0.75]

for feat in skewed.index:
    x = all_data[feat]
    # boxcox_normmax needs strictly positive input; shift by 1 for non-negative features
    if (x >= 0).all():
        try:
            lam = boxcox_normmax(x + 1)
            # the optimizer occasionally blows up on near-degenerate features
            if not np.isfinite(lam) or abs(lam) > 1.5:
                lam = 0.15
        except Exception:
            lam = 0.15
    else:
        lam = 0.15
    all_data[feat] = boxcox1p(x, lam)

all_data = pd.get_dummies(all_data)

# Drop one-hots with fewer than 5 nonzero rows — pure noise that hurts linear models
nonzero_counts = (all_data != 0).sum()
rare_cols = nonzero_counts[nonzero_counts < 5].index.tolist()
if rare_cols:
    all_data = all_data.drop(columns=rare_cols)

X_train = all_data.iloc[:ntrain, :].copy()
X_test  = all_data.iloc[ntrain:, :].copy()

X_train = pd.concat([all_ID[:ntrain].reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
X_test  = pd.concat([all_ID[ntrain:].reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

X_train.to_csv(rf'{OUT}\train_cleaned.csv', index=False)
X_test.to_csv(rf'{OUT}\test_cleaned.csv', index=False)
y_train.to_csv(rf'{OUT}\y_train.csv', index=False)

print("Train shape:", X_train.shape)
print("Test  shape:", X_test.shape)
print("Any NaNs?", X_train.isnull().any().any(), X_test.isnull().any().any())
numeric_train = X_train.select_dtypes(include=[np.number])
numeric_test  = X_test.select_dtypes(include=[np.number])
print("Any infinities?", np.isinf(numeric_train.values).any(), np.isinf(numeric_test.values).any())
print(f"Box-Cox transformed {len(skewed)} skewed features | dropped {len(rare_cols)} rare one-hots")
