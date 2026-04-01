import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# 1. ENCODING

def encode_data(train, test):
    
    # tách target ra trước
    y = train["SalePrice"]
    train = train.drop("SalePrice", axis=1)

    # one-hot encoding
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    # align feature space
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    # ghép lại target
    train["SalePrice"] = y

    return train,test

# 2. EVALUATION

def evaluate_model(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))

    return rmse

# 3. FEATURE IMPORTANCE

def plot_feature_importance(model, X):
    importance = model.feature_importances_
    feature_names = X.columns

    indices = np.argsort(importance)[-15:]

    plt.figure(figsize=(10,6))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top Feature Importances")
    plt.show()

# 4. TRAIN MODELS

def train_models(train):
    X = train.drop("SalePrice", axis=1)
    y = train["SalePrice"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    #  Linear Regression
    lr = LinearRegression()
    rmse_lr = evaluate_model(lr, X_train, X_val, y_train, y_val)
    results["LinearRegression"] = rmse_lr

    #  Random Forest (tuned)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rmse_rf = evaluate_model(rf, X_train, X_val, y_train, y_val)
    results["RandomForest"] = rmse_rf

    # XGBoost (tuned)
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    rmse_xgb = evaluate_model(xgb, X_train, X_val, y_train, y_val)
    results["XGBoost"] = rmse_xgb

    # Print results
    print("\nModel Performance (RMSE):")
    for model_name, score in results.items():
        print(f"{model_name}: {score:.5f}")

    # Feature importance (XGBoost)
    xgb.fit(X, y)
    plot_feature_importance(xgb, X)

    return xgb, results

# 5. CROSS VALIDATION

def cross_validate_model(train):
    X = train.drop("SalePrice", axis=1)
    y = train["SalePrice"]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        scoring='neg_mean_squared_error',
        cv=5
    )

    rmse_scores = np.sqrt(-scores)

    print("\nXGBoost CV RMSE:", rmse_scores.mean())

    return model

# 6. PREDICT TEST

def predict_test(model, test):
    preds = model.predict(test)
    preds = np.expm1(preds)  # inverse log

    return preds