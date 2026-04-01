from src.data_loader import load_data
from src.preprocesses import preprocess_data
from src.train import encode_data, train_models, cross_validate_model, predict_test

import pandas as pd


def main():
    # 1. LOAD DATA
    train, test = load_data()

    # 2. PREPROCESS
    train = preprocess_data(train, is_train=True)
    test = preprocess_data(test, is_train=False)

    # 3. ENCODING
    train, test = encode_data(train, test)

    # 4. TRAIN MODELS
    best_model, results = train_models(train)

    # 5. CROSS VALIDATION
    model = cross_validate_model(train)

    X = train.drop("SalePrice", axis=1)
    y = train["SalePrice"]
    model.fit(X, y)

    # 6. PREDICT TEST
    preds = predict_test(model, test)

    # 7. SAVE SUBMISSION
    submission = pd.read_csv("sample_submission.csv")
    submission["SalePrice"] = preds

    submission.to_csv("submission.csv", index=False)

    print("\nSubmission file created: submission.csv")


if __name__ == "__main__":
    main()