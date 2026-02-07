import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, LinearRegression

import xgboost as xgb

# =========================
# User parameters
# =========================
N_SPLITS = 14
RANDOM_STATE = 42

plot_dir = r'F:\housing_prices\plots\performance'
os.makedirs(plot_dir, exist_ok=True)

# =========================
# Load data
# =========================
X = pd.read_csv(r'F:\housing_prices\export\train_cleaned.csv')
X_test = pd.read_csv(r'F:\housing_prices\export\test_cleaned.csv')
y = pd.read_csv(r'F:\housing_prices\export\y_train.csv')['SalePrice']

X_id = X['Id']
X = X.drop('Id', axis=1)
X_test_id = X_test['Id']
X_test = X_test.drop('Id', axis=1)

feature_names = X.columns

# =========================
# Handle infinities / extreme values
# =========================
numeric_cols = X.select_dtypes(include=[np.number]).columns

# Replace inf/-inf with NaN
X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
X_test[numeric_cols] = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan)

# Fill NaNs with median (from training data)
for col in numeric_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Clip extreme values to avoid float64 issues
clip_value = 1e10
X[numeric_cols] = X[numeric_cols].clip(-clip_value, clip_value)
X_test[numeric_cols] = X_test[numeric_cols].clip(-clip_value, clip_value)

# =========================
# Metric
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =========================
# Models
# =========================
ridge = RidgeCV(alphas=np.logspace(-3, 2, 30))
lasso = LassoCV(alphas=np.logspace(-4, -1, 30), max_iter=15000)
linreg = LinearRegression()

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective='reg:squarederror',
    random_state=RANDOM_STATE
)

# =========================
# OOF predictions (all features)
# =========================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof = {
    'ridge': np.zeros(len(X)),
    'lasso': np.zeros(len(X)),
    'lin':   np.zeros(len(X)),
    'xgb':   np.zeros(len(X))
}

ridge_coef = np.zeros(len(feature_names))
lasso_coef = np.zeros(len(feature_names))
xgb_imp = np.zeros(len(feature_names))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # Scale inside fold (no leakage)
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    ridge.fit(X_tr_s, y_tr)
    lasso.fit(X_tr_s, y_tr)
    linreg.fit(X_tr_s, y_tr)
    xgb_model.fit(X_tr_s, y_tr)

    oof['ridge'][val_idx] = ridge.predict(X_val_s)
    oof['lasso'][val_idx] = lasso.predict(X_val_s)
    oof['lin'][val_idx]   = linreg.predict(X_val_s)
    oof['xgb'][val_idx]   = xgb_model.predict(X_val_s)

    ridge_coef += np.abs(ridge.coef_)
    lasso_coef += np.abs(lasso.coef_)
    xgb_imp += xgb_model.feature_importances_

# =========================
# OOF scores
# =========================
rmse_scores = {
    'Linear': rmse(y, oof['lin']),
    'Ridge':  rmse(y, oof['ridge']),
    'Lasso':  rmse(y, oof['lasso']),
    'XGB':    rmse(y, oof['xgb'])
}

print("\nOOF RMSE (all features):")
for k, v in rmse_scores.items():
    print(f"{k:<8}: {v:.5f}")

# =========================
# Feature importance (aggregated)
# =========================
ridge_imp = pd.Series(ridge_coef / N_SPLITS, index=feature_names)
lasso_imp = pd.Series(lasso_coef / N_SPLITS, index=feature_names)
xgb_imp = pd.Series(xgb_imp / N_SPLITS, index=feature_names)

# =========================
# Weighted averaging of OOF predictions
# =========================
rmse_values = np.array([rmse_scores['Ridge'], rmse_scores['Lasso'], rmse_scores['XGB']])
weights = 1 / rmse_values
weights /= weights.sum()  # normalize

oof_weighted = (
    weights[0] * oof['ridge'] +
    weights[1] * oof['lasso'] +
    weights[2] * oof['xgb']
)

weighted_rmse = rmse(y, oof_weighted)
print(f"\nOOF RMSE (weighted average): {weighted_rmse:.5f}")
print(f"Weights: Ridge={weights[0]:.3f}, Lasso={weights[1]:.3f}, XGB={weights[2]:.3f}")

# =========================
# Stacking using all features
# =========================
scaler_full = RobustScaler()
X_scaled = scaler_full.fit_transform(X)
X_test_scaled = scaler_full.transform(X_test)  # use transform only

oof_stack = np.column_stack([oof['ridge'], oof['lasso'], oof['xgb']])

meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_stack, y)

stack_rmse = rmse(y, meta_model.predict(oof_stack))
print(f"\nOOF RMSE (stacked all features): {stack_rmse:.5f}")

# =========================
# Fit base models on full data
# =========================
ridge.fit(X_scaled, y)
lasso.fit(X_scaled, y)
xgb_model.fit(X_scaled, y)

test_stack = np.column_stack([
    ridge.predict(X_test_scaled),
    lasso.predict(X_test_scaled),
    xgb_model.predict(X_test_scaled)
])

test_pred = meta_model.predict(test_stack)

# =========================
# Diagnostic plot
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(y, meta_model.predict(oof_stack), alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual SalePrice')
plt.ylabel('Stacked Prediction')
plt.title(f'Stacking (All Features) | RMSE={stack_rmse:.4f}')
plt.tight_layout()
plt.savefig('F:\housing_prices\plots\performance\stack_residuals.png', dpi=1200, bbox_inches='tight')
plt.show()