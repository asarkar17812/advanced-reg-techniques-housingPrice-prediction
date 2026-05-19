import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import lightgbm as lgb

N_SPLITS     = 14
RANDOM_STATE = 42
BAG_SEEDS    = (42, 1337, 2024)   # seed-bag the boosters to cut variance

X      = pd.read_csv(r'F:\housing_prices\export\train_cleaned.csv')
X_test = pd.read_csv(r'F:\housing_prices\export\test_cleaned.csv')
y      = pd.read_csv(r'F:\housing_prices\export\y_train.csv')['SalePrice']

X_id      = X['Id']; X      = X.drop('Id', axis=1)
X_test_id = X_test['Id']; X_test = X_test.drop('Id', axis=1)
feature_names = X.columns

# Defensive cleanup — preprocessing should already be clean
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols]      = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
X_test[numeric_cols] = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan)
for col in numeric_cols:
    med = X[col].median()
    X[col]      = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)
clip_value = 1e10
X[numeric_cols]      = X[numeric_cols].clip(-clip_value, clip_value)
X_test[numeric_cols] = X_test[numeric_cols].clip(-clip_value, clip_value)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ----- Base models -----
linreg = LinearRegression()
ridge  = RidgeCV(alphas=np.logspace(-3, 2, 30))
lasso  = LassoCV(alphas=np.logspace(-4, -1, 30), max_iter=15000, random_state=RANDOM_STATE)
enet   = ElasticNetCV(
    alphas=np.logspace(-4, -1, 30),
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
    max_iter=50000, random_state=RANDOM_STATE,
)
# KernelRidge with a degree-2 polynomial kernel = closed-form Ridge over all pairwise
# interactions. Catches the non-linear "quality × size" surface without an explicit
# polynomial expansion that would blow up feature count. The (alpha, coef0) here are
# the standard Ames values reported in several Kaggle write-ups.
kridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


def make_xgb(seed):
    return xgb.XGBRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_alpha=0.05,
        reg_lambda=1.0,
        gamma=0.0,
        objective='reg:squarederror',
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=100,
        eval_metric='rmse',
    )


def make_lgb(seed):
    return lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        num_leaves=16,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.4,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

MODEL_KEYS = ['ridge', 'lasso', 'enet', 'kridge', 'xgb', 'lgb', 'lin']
oof       = {k: np.zeros(len(X))      for k in MODEL_KEYS}
test_pred = {k: np.zeros(len(X_test)) for k in MODEL_KEYS}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # Scale inside each fold to keep validation stats out of the scaling fit
    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_test)

    ridge.fit(X_tr_s, y_tr)
    lasso.fit(X_tr_s, y_tr)
    enet.fit(X_tr_s, y_tr)
    linreg.fit(X_tr_s, y_tr)
    kridge.fit(X_tr_s, y_tr)

    oof['ridge'][val_idx]  = ridge.predict(X_val_s)
    oof['lasso'][val_idx]  = lasso.predict(X_val_s)
    oof['enet'][val_idx]   = enet.predict(X_val_s)
    oof['lin'][val_idx]    = linreg.predict(X_val_s)
    oof['kridge'][val_idx] = kridge.predict(X_val_s)

    test_pred['ridge']  += ridge.predict(X_te_s)  / N_SPLITS
    test_pred['lasso']  += lasso.predict(X_te_s)  / N_SPLITS
    test_pred['enet']   += enet.predict(X_te_s)   / N_SPLITS
    test_pred['lin']    += linreg.predict(X_te_s) / N_SPLITS
    test_pred['kridge'] += kridge.predict(X_te_s) / N_SPLITS

    # Seed-bag the boosters: average across BAG_SEEDS within each fold to reduce
    # the variance from random column/row subsampling. Each seed gets its own
    # early-stopping decision.
    xgb_val  = np.zeros(len(val_idx))
    lgb_val  = np.zeros(len(val_idx))
    xgb_test = np.zeros(len(X_test))
    lgb_test = np.zeros(len(X_test))
    for seed in BAG_SEEDS:
        xm = make_xgb(seed)
        xm.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        xgb_val  += xm.predict(X_val_s) / len(BAG_SEEDS)
        xgb_test += xm.predict(X_te_s)  / len(BAG_SEEDS)

        lm = make_lgb(seed)
        lm.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
               callbacks=[lgb.early_stopping(100, verbose=False)])
        lgb_val  += lm.predict(X_val_s) / len(BAG_SEEDS)
        lgb_test += lm.predict(X_te_s)  / len(BAG_SEEDS)

    oof['xgb'][val_idx] = xgb_val
    oof['lgb'][val_idx] = lgb_val
    test_pred['xgb'] += xgb_test / N_SPLITS
    test_pred['lgb'] += lgb_test / N_SPLITS


rmse_scores = {
    'Linear':      rmse(y, oof['lin']),
    'Ridge':       rmse(y, oof['ridge']),
    'Lasso':       rmse(y, oof['lasso']),
    'ElasticNet':  rmse(y, oof['enet']),
    'KernelRidge': rmse(y, oof['kridge']),
    'XGB':         rmse(y, oof['xgb']),
    'LGBM':        rmse(y, oof['lgb']),
}
print("\nOOF RMSE per base model:")
for k, v in rmse_scores.items():
    print(f"  {k:<11}: {v:.5f}")


# ----- Weighted average via constrained NNLS -----
STACK_KEYS = ['ridge', 'lasso', 'enet', 'kridge', 'xgb', 'lgb']
oof_stack  = np.column_stack([oof[k] for k in STACK_KEYS])
test_stack = np.column_stack([test_pred[k] for k in STACK_KEYS])

w_nnls, _ = nnls(oof_stack, y.values)
if w_nnls.sum() > 0:
    w_nnls = w_nnls / w_nnls.sum()
oof_blend  = oof_stack  @ w_nnls
test_blend = test_stack @ w_nnls
blend_rmse = rmse(y, oof_blend)

print(f"\nOOF RMSE (NNLS blend, {len(STACK_KEYS)} models): {blend_rmse:.5f}")
for k, w in zip(STACK_KEYS, w_nnls):
    print(f"  {k:<7}: weight = {w:.3f}")


# ----- Stacked Ridge meta-model -----
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_stack, y)
oof_meta   = meta_model.predict(oof_stack)
test_meta  = meta_model.predict(test_stack)
stack_rmse = rmse(y, oof_meta)
print(f"\nOOF RMSE (stacked Ridge meta-model): {stack_rmse:.5f}")
print("  meta coefs:", dict(zip(STACK_KEYS, np.round(meta_model.coef_, 3))))


# ----- Final test predictions: average the NNLS blend and the Ridge meta-model -----
final_test_log = 0.5 * test_blend + 0.5 * test_meta
final_test     = np.expm1(final_test_log)
final_oof_log  = 0.5 * oof_blend  + 0.5 * oof_meta
final_rmse     = rmse(y, final_oof_log)
print(f"\nOOF RMSE (final = 0.5*NNLS + 0.5*Ridge_meta): {final_rmse:.5f}")

submission = pd.DataFrame({'Id': X_test_id, 'SalePrice': final_test})
submission.to_csv(r'F:\housing_prices\export\submission.csv', index=False)
print(f"\nSubmission written -> export/submission.csv  (n={len(submission)})")


# ----- Diagnostic plot -----
plt.figure(figsize=(8, 6))
plt.scatter(y, final_oof_log, alpha=0.4, s=12)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=1)
plt.xlabel('Actual log(SalePrice)')
plt.ylabel('Final ensemble prediction')
plt.title(f'Ensemble (NNLS + Ridge meta avg) | OOF RMSE = {final_rmse:.4f}')
plt.tight_layout()
plt.savefig(r'F:\housing_prices\plots\performance\stack_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
