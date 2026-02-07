# Housing Prices Prediction

This repository contains a complete workflow for predicting house prices using the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The workflow includes **data preprocessing, feature engineering, and model stacking with weighted averaging**. 

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Data Preprocessing](#data-preprocessing)  
- [Modeling Architecture](#modeling-architecture)  
- [Usage](#usage)  
- [Results](#results)  
- [Dependencies](#dependencies)  

---

## Project Overview

The goal of this project is to predict the sale price of residential homes using a mix of numerical and categorical features. The workflow implements:

- Robust handling of missing and extreme values
- Feature engineering including total square footage, bathrooms, and age
- Skew correction for numerical features
- One-hot encoding for categorical features
- Out-of-Fold (OOF) predictions for validation
- Weighted averaging and stacking ensemble for improved accuracy

---

## Data Preprocessing

The preprocessing pipeline includes:

1. **Loading Data**
   - Training and test CSV files are loaded separately.
   
2. **Outlier Removal**
   - Extreme values in `GrLivArea` (>4000 sq ft) are removed to prevent skewed modeling.

3. **Target Transformation**
   - `SalePrice` is log-transformed to reduce skewness.

4. **Missing Values Handling**
   - Categorical features with missing values are filled with `'None'`.
   - Numerical features with missing values are filled with 0 or the median per neighborhood.
   - Garage year feature gets an indicator for missing garages.

5. **Feature Engineering**
   - Total square footage, total porch area, total bathrooms, age, remodel age, and presence indicators for garage, basement, and fireplace.
   - Interaction features like `GrLivArea_by_TotalSF` and `OverallQual_by_Age`.

6. **Encoding**
   - Ordinal mappings for quality-related features.
   - One-hot encoding for categorical variables.
   - Skewed numerical features are transformed using Box-Cox power transform.

7. **Data Cleaning**
   - Infinite values and extreme values are clipped to avoid numerical instability.

---

## Modeling Architecture

The modeling workflow implements a **stacked ensemble** of three base models:

1. **Ridge Regression** (`RidgeCV`)
2. **Lasso Regression** (`LassoCV`)
3. **XGBoost Regression** (`XGBRegressor`)

### Workflow:

1. **Out-of-Fold Predictions**
   - K-Fold cross-validation (14 folds) is used to generate OOF predictions for each base model.

2. **Weighted Averaging**
   - Base model predictions are combined using weights inversely proportional to each model’s OOF RMSE.

3. **Stacking**
   - OOF predictions from base models are used as features to train a Ridge regression meta-model.
   - This meta-model produces final predictions on the training set and test set.

4. **Scaling**
   - All features are scaled using `RobustScaler` to improve model robustness to outliers.

---