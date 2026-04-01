# House Price Prediction (End-to-End Machine Learning Project)

## Overview

This project develops a complete machine learning pipeline to predict house prices using structured data. The implementation follows a standard data science workflow, including data exploration, preprocessing, feature engineering, model training, evaluation, and prediction generation.

---

## Project Highlights

* Built an end-to-end machine learning pipeline from raw data to final predictions
* Applied domain-aware preprocessing to handle missing values correctly
* Engineered features to improve predictive performance
* Compared multiple models and optimized XGBoost through hyperparameter tuning
* Achieved RMSE ~0.12 on validation data
* Designed a modular and reproducible codebase using Python

---

## Dataset

* Source: Kaggle – House Prices (Advanced Regression Techniques)
* Training samples: 1460
* Test samples: 1459
* Features: 80+

---

## Pipeline Overview

```text id="pipeline1"
Raw Data
   ↓
Data Loading
   ↓
Exploratory Data Analysis (EDA)
   ↓
Data Cleaning (Missing Values, Outliers)
   ↓
Feature Engineering
   ↓
Encoding (One-Hot)
   ↓
Model Training (Linear, RF, XGBoost)
   ↓
Evaluation (RMSE, Cross Validation)
   ↓
Hyperparameter Tuning
   ↓
Final Model
   ↓
Prediction (Test Data)
   ↓
Submission File
```

---

## Pipeline Explanation

### 1. Data Loading

* Loaded training and test datasets from raw CSV files
* Maintained separation between train and test to avoid data leakage

### 2. Exploratory Data Analysis (EDA)

* Analyzed feature distributions and relationships with target
* Identified skewness in `SalePrice` and applied log transformation
* Distinguished between true missing values and structural absence

### 3. Data Cleaning

* Filled missing categorical features with "None" where absence is meaningful
* Used median imputation for skewed numerical variables
* Removed extreme outliers to improve model stability

### 4. Feature Engineering

* Created `TotalSF` to capture total living space
* Improved model signal by combining related features

### 5. Encoding

* Applied one-hot encoding to categorical variables
* Aligned train and test feature spaces to ensure consistency

### 6. Model Training

* Trained multiple models:

  * Linear Regression (baseline)
  * Random Forest
  * XGBoost

### 7. Evaluation

* Used RMSE as the main evaluation metric
* Applied train/validation split and cross-validation
* Ensured model generalization

### 8. Hyperparameter Tuning

* Tuned XGBoost parameters:

  * n_estimators
  * learning_rate
  * max_depth
  * subsample
  * colsample_bytree

### 9. Prediction

* Generated predictions on test dataset
* Applied inverse log transformation
* Exported results as submission file

---

## Results

| Model             | RMSE  |
| ----------------- | ----- |
| Linear Regression | ~0.13 |
| Random Forest     | ~0.14 |
| XGBoost           | ~0.12 |

XGBoost achieved the best performance after tuning.

---

## Feature Importance

The most influential features include:

* OverallQual
* TotalSF
* ExterQual
* GarageCars

These results indicate that construction quality and usable space are the primary drivers of house prices.

---

## Project Structure

```text id="structure1"
data/
├── raw/
├── processed_data/

src/
├── data_loader.py
├── eda.py
├── preprocesses.py
├── train.py

outputs/
├── feature_importance.png
├── submission.csv

main.py
requirements.txt
```

---

## How to Run

```bash id="run1"
pip install -r requirements.txt
python main.py
```

---

## Key Takeaways

* Missing data can represent real-world conditions, not just noise
* Feature engineering significantly impacts model performance
* Tree-based models outperform linear models on structured datasets
* Proper pipeline design ensures reproducibility and prevents data leakage

---

## Future Improvements

* Feature selection and dimensionality reduction
* Ensemble methods (stacking)
* Model explainability using SHAP

---

## Author

Tien Dat
