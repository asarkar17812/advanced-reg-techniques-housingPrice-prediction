# Housing Prices Prediction

This repository contains a complete workflow for predicting house prices using the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The workflow includes **data preprocessing, feature engineering, and model stacking** with both NNLS-blended and Ridge-meta ensembles.

I began this project in November 2024 and am revisiting it during an Intro to Machine Learning class. The current revision focuses on (a) fixing several preprocessing bugs in the original pipeline, (b) preserving ordinal information that was previously thrown away, (c) adding three more base learners (ElasticNet, LightGBM, KernelRidge) for ensemble diversity, (d) closing a leakage hole in the stacking step, and (e) seed-bagging the boosters to reduce variance. The combined effect is a ~2.3% drop in cross-validated RMSE relative to the prior version (0.10786 → 0.10537).

---

## Table of Contents

- [Results](#results)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Architecture](#modeling-architecture)
- [What Changed and Why](#what-changed-and-why)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Results

Out-of-fold (OOF) RMSE on `log1p(SalePrice)`, 14-fold cross validation, `random_state=42`. The "Previous" column is the OOF RMSE produced by the prior version of the pipeline on the same folds; "Current" is this revision.

| Model / Ensemble                           | Previous | Current           | Δ             |
| ------------------------------------------ | -------- | ----------------- | -------------- |
| Linear Regression                          | 0.12672  | 0.12234           | −0.00438      |
| Ridge Regression                           | 0.11126  | 0.11003           | −0.00123      |
| Lasso Regression                           | 0.10988  | 0.10818           | −0.00170      |
| ElasticNet                                 | —       | 0.10817           | —             |
| KernelRidge (poly-2)                       | —       | 0.11634           | —             |
| XGBoost (seed-bagged ×3)                  | 0.11783  | 0.10940           | −0.00843      |
| LightGBM (seed-bagged ×3)                 | —       | 0.11208           | —             |
| **NNLS Blend** (was: Weighted)       | 0.10809  | **0.10534** | −0.00275      |
| **Stacked Ridge Meta**               | 0.10786  | **0.10544** | −0.00242      |
| **Final** (0.5·NNLS + 0.5·Stacked) | —       | **0.10537** | **Best** |

**Final submission** is the average of the NNLS blend and the Ridge meta-model. Both ensembling strategies have nearly identical OOF RMSE but make different errors, so averaging them is robust against either being miscalibrated.

**NNLS blend weights** (the constrained solver assigns zero to redundant models — Ridge, Lasso, and LightGBM are all subsumed by the chosen four):

- ElasticNet: **0.472**
- XGBoost: **0.429**
- KernelRidge: **0.099**

**Stacked Ridge meta coefficients** (not constrained to sum to one; mild shrinkage toward zero):

- Ridge: 0.033
- Lasso: 0.222
- ElasticNet: 0.223
- KernelRidge: 0.122
- XGBoost: 0.306
- LightGBM: 0.102

![Stacked predictions vs. actual log(SalePrice)](plots/performance/stack_residuals.png)

---

## Project Overview

The goal is to predict the sale price of residential homes from a mix of 79 numerical and categorical features describing each property. The workflow:

- Robust handling of missing values driven by what each `NA` actually means in `data_description.txt` (e.g. `PoolQC=NA` means *no pool*, not *unknown pool*)
- Feature engineering: total square footage, total bath count, age and remodel-age, presence indicators, and quality × size interactions
- Per-feature Box-Cox skew correction on continuous numeric features only
- Ordinal encoding for naturally ordered categoricals; one-hot for the rest
- Six base learners spanning three model families: regularised linear (Ridge, Lasso, ElasticNet), kernel methods (KernelRidge with a degree-2 polynomial kernel), and gradient boosting (XGBoost, LightGBM — each seed-bagged ×3)
- 14-fold cross-validated out-of-fold predictions used both for unbiased model evaluation and as inputs to the meta-model
- Two complementary ensemble strategies (NNLS blend + Ridge stacker), averaged for the final test prediction

---

## Data Preprocessing

The pipeline lives in [`scripts/data_exploration.py`](scripts/data_exploration.py). Steps, with the rationale for each:

### 1. Outlier removal

Two rows with `GrLivArea > 4000 sq ft` are dropped, per the dataset author's [recommendation](http://jse.amstat.org/v19n3/decock.pdf). These are partially sold properties whose actual prices don't reflect typical market behavior.

### 2. Target transformation

`SalePrice` is right-skewed (skew ≈ 1.88). We apply `log1p`, which:

- Makes the residuals roughly Gaussian (linear models assume this)
- Turns RMSE on the transformed target into RMSLE on the original scale — exactly the Kaggle competition metric

### 3. Missing value imputation

`NaN` is not always "unknown". `data_description.txt` documents that for many features, `NaN` literally means *the property does not have this feature* (no pool, no garage, no fireplace, etc.). The pipeline classifies columns into four imputation strategies:

| Strategy             | Columns                                                                                                                                                                                                                | Why                                                                                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Fill with `'None'` | `Alley`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1/2`, `Fence`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `MasVnrType`, `MiscFeature`, `PoolQC` | `NaN` is semantic ("no basement", "no pool") — encoding it as a separate category keeps the information.                                            |
| Fill with `0`      | `BsmtFinSF1/2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`, `GarageArea`, `GarageCars`, `MasVnrArea`                                                                                   | Same logic for numeric size/count columns — no basement = 0 sq ft.                                                                                    |
| Group-median         | `LotFrontage`                                                                                                                                                                                                        | Lot frontage is missing for ~18% of rows and varies dramatically by neighborhood; group-by-`Neighborhood` median is the sharpest non-leaky estimate. |
| Mode                 | `MSZoning`, `Electrical`, `KitchenQual`, `Exterior1st`, `Exterior2nd`, `SaleType`                                                                                                                          | True random missingness in 1–4 rows each; mode is harmless.                                                                                           |
| Domain default       | `Functional` → `'Typ'`                                                                                                                                                                                            | The data description explicitly says to assume `Typ` if missing.                                                                                     |

`GarageYrBlt` also gets a `GarageYrBlt_missing` indicator column before being filled with 0 — "no garage" and "garage built in year 0" should not look identical to the model.

### 4. Feature engineering

Aggregates that consistently rank near the top in Ames feature-importance studies:

| Feature                                                            | Formula                                                                |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| `TotalSF`                                                        | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF`                                  |
| `TotalPorchSF`                                                   | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF` |
| `TotalSqFoot`                                                    | `TotalSF + TotalPorchSF`                                             |
| `TotalBath`                                                      | `BsmtFullBath + 0.5·BsmtHalfBath + FullBath + 0.5·HalfBath`        |
| `Age`                                                            | `clip(YrSold − YearBuilt, 0, ∞)`                                   |
| `RemodAge`                                                       | `clip(YrSold − YearRemodAdd, 0, ∞)`                                |
| `IsNew`                                                          | `1` if `Age == 0`                                                  |
| `IsRemodeled`                                                    | `1` if `YearBuilt ≠ YearRemodAdd`                                 |
| `HasGarage / HasBasement / HasFireplace / HasPool / Has2ndFloor` | binary presence indicators                                             |

The `clip(…, 0, ∞)` on age matters: the test set has rows with `YrSold < YearBuilt` (data-entry artefacts). Without clipping, the downstream `OverallQual_by_Age = OverallQual / (Age + 1)` division produces ±∞ when `Age == −1`.

### 5. Ordinal encoding

A major change from the prior version: ten categoricals with natural orderings are now mapped to integers rather than one-hot encoded.

| Column                                                                                                                                                | Ordering (low → high)                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `ExterQual`, `ExterCond`, `HeatingQC`, `KitchenQual`, `BsmtQual`, `BsmtCond`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC` | `None < Po < Fa < TA < Gd < Ex`                     |
| `BsmtExposure`                                                                                                                                      | `None < No < Mn < Av < Gd`                          |
| `BsmtFinType1`, `BsmtFinType2`                                                                                                                    | `None < Unf < LwQ < Rec < BLQ < ALQ < GLQ`          |
| `GarageFinish`                                                                                                                                      | `None < Unf < RFn < Fin`                            |
| `Fence`                                                                                                                                             | `None < MnWw < GdWo < MnPrv < GdPrv`                |
| `Functional`                                                                                                                                        | `Sal < Sev < Maj2 < Maj1 < Mod < Min2 < Min1 < Typ` |
| `PavedDrive`                                                                                                                                        | `N < P < Y`                                         |
| `LandSlope`                                                                                                                                         | `Sev < Mod < Gtl`                                   |
| `LotShape`                                                                                                                                          | `IR3 < IR2 < IR1 < Reg`                             |

`CentralAir` is collapsed to a 0/1 binary.

One-hot encoding throws this ordering away — Ridge and Lasso then have to relearn that `Gd > TA > Fa` from the data, which they do *poorly* on the smaller dummy columns. Ordinal encoding preserves it for free.

### 6. Quality × size interactions

The single strongest signal in Ames is that *good houses cost more per square foot* — a non-linearity that linear base models cannot express without explicit interactions. The added features:

- `OverallQual_x_TotalSF`
- `OverallQual_x_GrLivArea`
- `Qual_x_Cond` (`OverallQual × OverallCond`)
- `ExterQual_x_TotalSF`
- `KitchenQual_x_TotalSF`
- `BsmtQual_x_TotalBsmtSF`

### 7. Per-feature Box-Cox skew correction

Skewed continuous features (`|skew| > 0.75`) get Box-Cox transformed. The prior pipeline used a fixed `λ = 0.15` for every feature; this version calls `scipy.stats.boxcox_normmax` per feature to find the λ that maximises log-likelihood, with a fall-back to 0.15 if the optimiser fails or returns a degenerate value (`|λ| > 1.5`).

Critically, the skew-correction loop now excludes:

- Ordinal columns (they're integer grades, not continuous)
- Binary indicator columns (`HasGarage`, `IsNew`, etc. — Box-Cox-ing a 0/1 is meaningless)
- `OverallQual`, `OverallCond` (1–10 grade scales, not continuous quantities)

The prior pipeline was Box-Cox-ing all of these, which both wasted transforms and quietly introduced floating-point noise into indicator columns.

### 8. Rare-category cleanup

After `pd.get_dummies`, any one-hot column with fewer than 5 non-zero rows is dropped. These are effectively noise for linear models (a single coefficient fit to 1–2 observations) and the OHE explosion of `Exterior2nd`, `Neighborhood`, etc. produces dozens of them.

Final design matrix: **254 features** (down from 302 in the prior version, despite adding new engineered features — most of the reduction is from collapsing ordinals out of OHE and dropping rare dummies).

---

## Modeling Architecture

The pipeline lives in [`scripts/models.py`](scripts/models.py).

### Base learners

| Model        | Role in the ensemble                                                                                                                                                                                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear (OLS) | Reference only; never enters the ensemble. Useful as a sanity floor — if a regularised model performs worse than OLS, something is wrong.                                                                                                         |
| Ridge        | L2 penalty; handles correlated features (and we have many:`TotalSF` ≡ `TotalBsmtSF + 1stFlrSF + 2ndFlrSF`).                                                                                                                                   |
| Lasso        | L1 penalty; sparsifies coefficients. Drives noise dummies to zero entirely, which Ridge can't do.                                                                                                                                                  |
| ElasticNet   | L1 + L2 blend (`l1_ratio ∈ {0.1, 0.5, 0.7, 0.9, 0.95, 1.0}`). Handles **groups** of correlated features (where Lasso arbitrarily picks one and zeros the rest).                                                                           |
| KernelRidge  | Polynomial kernel of degree 2 (`alpha=0.6, coef0=2.5`). Closed-form Ridge over *all* pairwise feature interactions — captures the non-linear quality × size surface without the feature-count explosion of an explicit polynomial expansion. |
| XGBoost      | Captures non-linearities and high-order interactions the linear stack can't. Early stopping with 100-round patience;**seed-bagged across 3 seeds** to cut variance.                                                                          |
| LightGBM     | Independent gradient-boosting implementation; uses leaf-wise growth instead of XGB's level-wise. Adds genuine diversity to the ensemble. Also seed-bagged ×3.                                                                                     |

All boosters use `early_stopping_rounds=100` against the fold's validation set, with `n_estimators=4000` as a ceiling. The early stopping decision is made independently for each of the three bag seeds. Seed-bagging adds roughly 3× the booster training cost but removes most of the run-to-run RMSE jitter caused by row/column subsampling — XGB's OOF RMSE dropped from 0.11158 (single seed) to 0.10940 (3-seed bag) on otherwise identical settings.

### Cross-validation

14-fold KFold with `shuffle=True, random_state=42`. Inside each fold:

1. `RobustScaler` is `fit_transform`-ed on the train slice and `transform`-ed on the val and test slices. Fitting the scaler inside the fold (not on the full training set) avoids leakage of validation statistics into the scaling.
2. All seven base models are trained on the train slice (boosters are trained three times per fold, once per bag seed, and their predictions are averaged).
3. Each model predicts on the val slice → OOF prediction for the held-out rows.
4. Each model predicts on the **test slice** → fold-test prediction. The 14 fold-test predictions are averaged at the end.

### Stacking leakage fix

The prior pipeline:

1. Trained base models in 14 folds → OOF predictions.
2. Fit a Ridge meta-model on OOF stack.
3. **Refit base models on the full training set** to produce test predictions.
4. Fed those test predictions to the meta-model.

The OOF predictions in step 2 are *held-out* — they're noisy estimates of what the base models would say about unseen data. The full-refit test predictions in step 3 are *not* held-out — they're tighter, with different bias/variance. The meta-model was therefore trained on one distribution and applied to another.

The current pipeline averages the 14 fold-test predictions (step 4 in the bullet list above). Each fold-test prediction *is* held-out (the base model never saw the test set during training, and the fold's training data is only 13/14 of the full train), so the statistical character matches the OOF predictions the meta-model was trained on.

### Ensembling

Two complementary strategies:

1. **NNLS blend**: solve `min_w ‖ OOF·w − y ‖` s.t. `w ≥ 0`. Constrained optimal weights — no model can drag the blend down by getting a negative coefficient. In this run NNLS allocated weight to ElasticNet (0.472), XGBoost (0.429), and KernelRidge (0.099); the other three models were redundant given those three and got zero weight.
2. **Stacked Ridge meta-model**: fit `Ridge(alpha=1.0)` on the OOF predictions stacked as features. Allows negative coefficients (in principle — they don't appear here) and applies mild shrinkage, which often helps when OOF columns are highly correlated. Ridge meta puts non-zero weight on all six models.

Final test prediction = `0.5 · NNLS_blend + 0.5 · Ridge_meta`, then `expm1` to invert the log target. Saved to [`export/submission.csv`](export/submission.csv).

---

## What Changed and Why

A summary of the diffs from the prior pipeline, ordered roughly by impact on OOF RMSE.

### Preprocessing

1. **Fixed `Age = -1 → ÷0 → Inf` in test set.** Some test rows had `YrSold < YearBuilt`. Clipping `Age` and `RemodAge` to ≥ 0 fixed the infinities that the modeling script was silently masking with a 1e10 clip.
2. **Stopped Box-Cox-ing indicator and ordinal columns.** The prior script computed skew on every numeric column including `HasGarage`, `GarageYrBlt_missing`, `OverallQual`, etc. and Box-Cox'd anything with `|skew| > 0.75`. This produced fractional values in columns that were supposed to be 0/1 grades. Excluding those columns from the skew calc is both more principled and slightly improves linear-model RMSE.
3. **Ordinal mappings for 10 naturally ordered categoricals.** Previously all of `BsmtExposure`, `BsmtFinType1/2`, `GarageFinish`, `Fence`, `Functional`, `PavedDrive`, `LandSlope`, `LotShape` were one-hot encoded — discarding the ordering and forcing the linear models to learn one coefficient per level from a handful of rows. Ordinal integer mapping preserves the ordering for free.
4. **Quality × size interactions.** Six new features. Linear models cannot express "good houses cost more per square foot" without an explicit `Qual × SF` term. These features show up in the Lasso top-20 in the new pipeline.
5. **Per-feature Box-Cox λ.** `boxcox_normmax` per skewed feature instead of a fixed 0.15. Small effect on linear models, harmless for trees.
6. **Drop rare one-hot columns (< 5 nonzero rows).** Lasso would mostly zero these anyway, but Ridge wouldn't, and they're pure noise for both. Dropped 21 such columns.
7. **`WoodDeckSF` added to `TotalPorchSF`.** Was previously a stand-alone column even though it's the same kind of outdoor square footage as the four porches.

### Modeling

8. **Stacking leakage fix** (per-fold test predictions averaged across folds, instead of refitting base models on the full train set and predicting test). This is the single most-defended change in the diff: the prior pattern is a well-known stacking anti-pattern that systematically miscalibrates the meta-model.
9. **Added ElasticNet.** Lasso vs. ElasticNet OOF RMSE is essentially tied (0.10818 vs 0.10817), but the NNLS blend prefers ElasticNet because it correlates less with XGBoost — `l1_ratio < 1` keeps small but nonzero weights on groups of correlated features that Lasso zeroes.
10. **Added LightGBM.** Independent GBM implementation that uses leaf-wise (not level-wise) growth. Slightly worse than XGB on its own (0.11208 vs 0.10940 seed-bagged) but the Ridge meta-model picks up 10% weight on it, which means it's adding information that the other models miss.
11. **Added KernelRidge with a degree-2 polynomial kernel.** Catches the non-linear "quality × size" surface implicitly via the kernel trick — every pairwise product of (scaled) features appears in the model without ever being materialised. On its own it's the weakest base learner (0.11634) but NNLS still assigns it ~10% weight and the Ridge meta gives it 12%, because its errors are uncorrelated with the linear models'. This is exactly what you want from an ensemble member: doesn't need to be the best, needs to be *different*.
12. **Seed-bagging the boosters.** Each fold trains XGB and LGBM three times with seeds `(42, 1337, 2024)` and averages the predictions. XGB's OOF RMSE went from 0.11158 → 0.10940 just from this change; LGBM from 0.11309 → 0.11208. Boosters with column/row subsampling have meaningful seed variance on small datasets, and bagging is the cheapest way to wash it out.
13. **Early stopping for XGB and LightGBM.** Tree count is no longer a hyperparameter; the model picks it per fold (and per bag seed) based on the validation curve. Also re-tuned XGB: `max_depth 6 → 3`, `colsample_bytree 0.8 → 0.4`, `min_child_weight 5 → 2`. Smaller trees + heavier column subsampling regularises better on a 1456-row dataset, and the seed bag papers over the noise from `colsample=0.4`.
14. **NNLS for blend weights** instead of inverse-RMSE heuristic. The prior pipeline weighted Ridge/Lasso/XGB by `1/RMSE` normalised — a reasonable heuristic but provably suboptimal. NNLS solves the actual constrained least-squares problem and can drop redundant models to zero (which is what happens here for Ridge, Lasso, and LightGBM in the NNLS blend — they're highly correlated with their cousins).
15. **Average of NNLS blend + Ridge meta** for the final test prediction. Both ensemble strategies have very similar OOF RMSE but make different errors; averaging reduces variance. The averaged ensemble (0.10537) edges out either strategy alone (0.10534 NNLS, 0.10544 Ridge meta).
16. **Save submission.csv.** The prior pipeline computed `test_pred` but never wrote it anywhere, so the result of an entire training run was lost.
17. **Plot dpi: 1200 → 150.** The diagnostic scatter was a 3.3 MB PNG. 150 dpi is still publication-grade and is 25× smaller.

---

## Usage

```bash
# 1. Preprocess
python scripts/data_exploration.py

# 2. Train ensemble, write OOF metrics, save submission.csv and the residuals plot
python scripts/models.py
```

Outputs:

- `export/train_cleaned.csv`, `export/test_cleaned.csv`, `export/y_train.csv` — design matrices
- `export/submission.csv` — final blended predictions on the Kaggle test set
- `plots/performance/stack_residuals.png` — diagnostic actual-vs-predicted scatter

---

## Dependencies

- `numpy`, `pandas`, `scipy`, `matplotlib`
- `scikit-learn` (RidgeCV, LassoCV, ElasticNetCV, KernelRidge, KFold, RobustScaler)
- `xgboost`
- `lightgbm`

Tested with scikit-learn 1.6.1, xgboost 3.1.3, lightgbm 4.6.0, numpy 2.2.x, pandas 2.2.x, scipy 1.15.x.
