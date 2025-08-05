from microimpute.models import QRF as MicroImputeQRF
import pandas as pd
import numpy as np
import pickle


class QRF:
    def __init__(self, file_path: str = None):
        if file_path is None:
            self.model = MicroImputeQRF()
        else:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.input_columns = data["input_columns"]

    def fit(self, X, y):
        train_df = pd.concat([X, y], axis=1)
        X_cols = X.columns
        y_cols = y.columns
        self.model = self.model.fit(train_df, X_cols, y_cols)
        self.input_columns = X.columns

    def predict(self, X):
        return self.model.predict(X)[0.5]

    def save(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(
                {"model": self.model, "input_columns": self.input_columns}, f
            )
