"""Baseline Random Forest model for engagement prediction."""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor


class BaselineModel:
    """Random Forest model for Instagram engagement prediction.

    This is our baseline model using only simple features.
    """

    def __init__(self, **model_params):
        """Initialize baseline model.

        Args:
            **model_params: Parameters for RandomForestRegressor
        """
        self.model = RandomForestRegressor(**model_params)
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model.

        Args:
            X: Training features
            y: Training target (likes)
        """
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features for prediction

        Returns:
            Array of predicted likes

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.

        Returns:
            DataFrame with features and importance scores

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, filepath: Path):
        """Save model to disk.

        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: Path) -> 'BaselineModel':
        """Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded BaselineModel instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        instance = cls()
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']

        return instance

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.model.get_params()

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_names) if self.feature_names else 0
        return f"BaselineModel({status}, n_features={n_features})"
