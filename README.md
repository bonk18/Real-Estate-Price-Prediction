# House Price Prediction: Machine Learning Project

This project aims to predict house prices using a combination of data preprocessing, feature engineering, and machine learning models. It utilizes the Ames Housing dataset as a basis and incorporates a variety of techniques to handle missing data, encode categorical features, scale numerical features, and address outliers.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Dataset Description](#dataset-description)
4. [Workflow](#workflow)
5. [Key Steps in Data Preprocessing](#key-steps-in-data-preprocessing)
6. [Machine Learning Models Used](#machine-learning-models-used)
7. [Results](#results)
8. [How to Run](#how-to-run)
9. [Future Enhancements](#future-enhancements)

---

## Project Overview
 The primary objective is to predict the `SalePrice` of houses using the provided features, such as square footage, the year built, and the number of rooms.

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - Data Manipulation: `pandas`, `numpy`
  - Data Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`

---

## Dataset Description
The dataset contains information about properties and their corresponding house prices (`SalePrice`). It includes:
- Numerical Features: e.g., `1stFlrSF`, `LotArea`
- Categorical Features: e.g., `Neighborhood`, `HouseStyle`
- Target Variable: `SalePrice`

### Key Columns:
- **`1stFlrSF` and `2ndFlrSF`:** Combined to calculate total living area.
- **`TotRmsAbvGrd`:** Total rooms above grade.
- **`YearBuilt`:** The construction year of the house.
- **`Utilities`:** Encoded to handle categorical data.
- **`SalePrice`:** The target variable for prediction.

---

## Workflow
1. **Data Cleaning and Preprocessing**
   - Handling missing values
   - Addressing outliers
   - Encoding categorical variables
   - Feature scaling
2. **Feature Engineering**
   - Adding a new feature: `TotalLivingArea` (`1stFlrSF + 2ndFlrSF`)
   - Adding another new feature `HouseAge`(`Year sold - Year built`)
   - Dropping less relevant features (e.g., features with minimal correlation to `SalePrice`)
3. **Modeling**
   - Splitting data into training and testing sets
   - Training multiple regression models
   - Hyperparameter tuning with `GridSearchCV`
4. **Evaluation**
   - Comparing models using metrics such as `R²`, `RMSE`, and `MAE`
   - Selecting the best-performing model

---

## Key Steps in Data Preprocessing

### 1. Handling Missing Data
- Imputed missing values 
- Dropped columns with excessive missing values, like `PoolQC` and `PoolArea`, when they had minimal impact on predictions.

### 2. Encoding Categorical Features
- Used `LabelEncoder` for label encoding.
- Applied consistent encoding for both training and testing datasets to avoid unseen labels.

### 3. Scaling Numerical Features
- Used `RobustScaler` to scale integer columns, reducing the influence of outliers.

### 4. Outlier Handling
- Applied capping at the 5th and 95th percentiles for relevant columns.

---

## Machine Learning Models Used
The following regression models were implemented:
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Decision Tree Regressor**
5. **Random Forest Regressor**
6. **Gradient Boosting Regressor**
7. **XGBoost Regressor**

### Model Evaluation Metrics
- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **Root Mean Squared Error (RMSE):** Penalizes larger errors more than MAE.
- **R² Score:** Represents the proportion of variance explained by the model.

---

## Results
| Model                  | MAE   | RMSE  | R²    |
|------------------------|-------|-------|-------|
| Linear Regression      | 21428.738455  | 41295.815478  | 0.755577  |
| Ridge Regression       | 21419.279985  | 41300.761354  | 0.755518  |
| Lasso Regression       | 21425.735972 | 41295.262190  | 0.755583  |
| Decision Tree          | 28639.632479  | 49941.944250  | 0.642512  |
| Random Forest          | 19331.947692  | 37831.965352  | 0.794861  |
| Gradient Boosting      | 17912.717008  | 37512.112590  | 0.798315  |
| XGBoost                | 18762.743222  | 32941.491830  | 0.844469  |

---
