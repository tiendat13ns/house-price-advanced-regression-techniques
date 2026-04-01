import pandas as pd

def load_data():
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    return train,test