"""
Quantile regression forest wrapper for imputation tasks.

This module provides a simplified interface to the microimpute QRF model
with serialisation support for trained models.
"""

from microimpute.models import QRF as MicroImputeQRF
import pandas as pd
import numpy as np
import pickle


class QRF:
    """
    Quantile regression forest model wrapper.

    Provides a simple interface to train and use quantile regression forests
    for imputation, with support for saving and loading trained models.
    """

    def __init__(self, file_path: str = None):
        """
        Initialise QRF model.

        Args:
            file_path: Path to saved model file. If None, creates new model.
        """
        if file_path is None:
            self.model = MicroImputeQRF()
        else:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.input_columns = data["input_columns"]

    def fit(self, X, y):
        """
        Train the model on input data.

        Args:
            X: Feature variables DataFrame.
            y: Target variables DataFrame.
        """
        train_df = pd.concat([X, y], axis=1)
        X_cols = X.columns
        y_cols = y.columns
        self.model = self.model.fit(train_df, X_cols, y_cols)
        self.input_columns = X.columns

    def predict(self, X):
        """
        Predict using the trained model.

        Args:
            X: Feature variables DataFrame.

        Returns:
            Predictions at the 0.5 quantile (median).
        """
        return self.model.predict(X)[0.5]

    def save(self, file_path: str):
        """
        Save trained model to file.

        Args:
            file_path: Path where model should be saved.
        """
        with open(file_path, "wb") as f:
            pickle.dump(
                {"model": self.model, "input_columns": self.input_columns}, f
            )
