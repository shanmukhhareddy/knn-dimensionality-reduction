import pandas as pd
from sklearn.model_selection import train_test_split

def get_data_split():
    df = pd.read_csv("data/processed_data.csv")

    X = df.drop("Status", axis=1)
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
