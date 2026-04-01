import pandas as pd
import numpy as np

# 1. HANDLE MISSING VALUES
def fill_missing(df):
    df = df.copy()

    # Missing = "None" (không tồn tại)
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2"
    ]

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numerical → median
    num_cols = ["LotFrontage", "MasVnrArea"]

    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical → mode
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    # Fix MasVnrType
    if "MasVnrType" in df.columns:
        df["MasVnrType"] = df["MasVnrType"].fillna("None")

    # Fix GarageYrBlt
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    return df

# 2. DROP USELESS FEATURES
def drop_features(df):
    df = df.copy()

    drop_cols = ["Id", "MiscFeature"]

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df

# 3. HANDLE OUTLIERS
def remove_outliers(df):
    df = df.copy()

    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        df = df[df["GrLivArea"] < 4000]

    return df

# 4. TRANSFORM TARGET
def transform_target(df):
    df = df.copy()

    if "SalePrice" in df.columns:
        df["SalePrice"] = np.log1p(df["SalePrice"])

    return df

# 5. FEATURE ENGINEERING
def create_features(df):
    df = df.copy()

    # tổng diện tích
    if all(col in df.columns for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    return df
# 6. FULL PIPELINE (CLEAN)
def preprocess_data(df, is_train=True):
    df = fill_missing(df)
    df = drop_features(df)

    if is_train:
        df = remove_outliers(df)
        df = transform_target(df)

    df = create_features(df)

    return df