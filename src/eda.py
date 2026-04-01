import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def basic_info(df):
    print(df.shape)
    print(df.info())
    print(df.describe())

def check_missing(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0])

def plot_target(df):
    sns.histplot(df['SalePrice'], kde = True)