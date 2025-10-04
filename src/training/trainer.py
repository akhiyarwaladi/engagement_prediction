#!/usr/bin/env python3
"""
Model training framework
Handles training, evaluation, and model persistence
"""

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

from src.training.experiment import ExperimentConfig, ExperimentResult, get_tracker


class ModelTrainer:
    """Train and evaluate models"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
        self.pca_dict = {}
        self.model = None
        self.feature_names = None

    def preprocess(self, X: pd.DataFrame, y: np.ndarray,
                  pca_components: Optional[Dict[str, int]] = None,
                  fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features

        Args:
            X: Feature matrix
            y: Target variable
            pca_components: Dict mapping feature prefix to n_components
                           e.g., {'bert_': 100, 'vit_': 100}
            fit: Whether to fit transformers

        Returns:
            X_processed, y_processed
        """

        X = X.copy()
        self.feature_names = X.columns.tolist()

        # Outlier clipping (99th percentile)
        if fit:
            self.y_99 = np.percentile(y, 99)
        y_clipped = np.clip(y, 0, self.y_99)

        # Log transformation
        y_log = np.log1p(y_clipped)

        # PCA for embeddings (if specified)
        if pca_components:
            X_parts = []
            processed_cols = set()

            for prefix, n_components in pca_components.items():
                # Find columns with this prefix
                cols = [c for c in X.columns if c.startswith(prefix)]

                if cols:
                    if fit:
                        pca = PCA(n_components=n_components, random_state=self.random_state)
                        X_transformed = pca.fit_transform(X[cols])
                        self.pca_dict[prefix] = pca
                        var_explained = pca.explained_variance_ratio_.sum()
                        print(f"  PCA {prefix}: {len(cols)} -> {n_components} ({var_explained*100:.1f}% variance)")
                    else:
                        pca = self.pca_dict[prefix]
                        X_transformed = pca.transform(X[cols])

                    X_parts.append(X_transformed)
                    processed_cols.update(cols)

            # Add remaining columns (not PCA'd)
            remaining_cols = [c for c in X.columns if c not in processed_cols]
            if remaining_cols:
                X_parts.append(X[remaining_cols].values)

            X_combined = np.hstack(X_parts) if X_parts else X.values
        else:
            X_combined = X.values

        # Quantile transformation
        if fit:
            self.scaler = QuantileTransformer(output_distribution='normal',
                                             random_state=self.random_state)
            X_scaled = self.scaler.fit_transform(X_combined)
        else:
            X_scaled = self.scaler.transform(X_combined)

        return X_scaled, y_log

    def train(self, X: pd.DataFrame, y: np.ndarray,
             model_type: str = 'stacking',
             model_params: Optional[Dict[str, Any]] = None,
             pca_components: Optional[Dict[str, int]] = None,
             test_size: float = 0.2) -> Dict[str, float]:
        """
        Train model

        Args:
            X: Feature matrix
            y: Target variable
            model_type: 'rf', 'hgb', 'stacking', 'stacking_gb'
            model_params: Model hyperparameters
            pca_components: PCA configuration
            test_size: Test set size

        Returns:
            Dict of metrics
        """

        print(f"\n[TRAIN] Training {model_type} model...")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {len(X)}")

        # Split data (time-based)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        # Preprocess
        print(f"\n[PREPROCESS] Applying transformations...")
        X_train_proc, y_train_log = self.preprocess(X_train, y_train, pca_components, fit=True)
        X_test_proc, y_test_log = self.preprocess(X_test, y_test, pca_components, fit=False)

        print(f"  Processed features: {X_train_proc.shape[1]}")

        # Create model
        model_params = model_params or {}
        self.model = self._create_model(model_type, model_params)

        # Train
        print(f"\n[FIT] Training model...")
        start_time = time.time()
        self.model.fit(X_train_proc, y_train_log)
        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate
        print(f"\n[EVALUATE] Computing metrics...")
        metrics = self._evaluate(X_train_proc, y_train, X_test_proc, y_test)

        metrics['train_time'] = train_time

        return metrics

    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance"""

        if model_type == 'rf':
            return RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                **params
            )

        elif model_type == 'hgb':
            return HistGradientBoostingRegressor(
                random_state=self.random_state,
                **params
            )

        elif model_type == 'stacking':
            # Default stacking with Ridge meta-learner
            rf = RandomForestRegressor(
                n_estimators=params.get('rf_n_estimators', 300),
                max_depth=params.get('rf_max_depth', 26),
                min_samples_split=params.get('rf_min_samples_split', 2),
                min_samples_leaf=params.get('rf_min_samples_leaf', 2),
                max_features=params.get('rf_max_features', 'sqrt'),
                random_state=self.random_state,
                n_jobs=-1
            )

            hgb = HistGradientBoostingRegressor(
                max_iter=params.get('hgb_max_iter', 254),
                max_depth=params.get('hgb_max_depth', 15),
                learning_rate=params.get('hgb_learning_rate', 0.104),
                min_samples_leaf=params.get('hgb_min_samples_leaf', 9),
                l2_regularization=params.get('hgb_l2', 0.146),
                random_state=self.random_state
            )

            return StackingRegressor(
                estimators=[('rf', rf), ('hgb', hgb)],
                final_estimator=Ridge(alpha=1.0),
                cv=5
            )

        elif model_type == 'stacking_gb':
            # Stacking with GradientBoosting meta-learner
            rf = RandomForestRegressor(
                n_estimators=params.get('rf_n_estimators', 300),
                max_depth=params.get('rf_max_depth', 26),
                random_state=self.random_state,
                n_jobs=-1
            )

            hgb = HistGradientBoostingRegressor(
                max_iter=params.get('hgb_max_iter', 254),
                max_depth=params.get('hgb_max_depth', 15),
                learning_rate=params.get('hgb_learning_rate', 0.104),
                random_state=self.random_state
            )

            gb_meta = GradientBoostingRegressor(
                n_estimators=params.get('meta_n_estimators', 100),
                max_depth=params.get('meta_max_depth', 5),
                learning_rate=params.get('meta_learning_rate', 0.05),
                random_state=self.random_state
            )

            return StackingRegressor(
                estimators=[('rf', rf), ('hgb', hgb)],
                final_estimator=gb_meta,
                cv=5
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on train and test sets"""

        # Predictions (in log space)
        y_train_pred_log = self.model.predict(X_train)
        y_test_pred_log = self.model.predict(X_test)

        # Transform back to original space
        y_train_pred = np.expm1(y_train_pred_log)
        y_test_pred = np.expm1(y_test_pred_log)

        # Metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        }

        # Print
        print(f"  Train: MAE={metrics['train_mae']:.2f}, R²={metrics['train_r2']:.3f}")
        print(f"  Test:  MAE={metrics['test_mae']:.2f}, R²={metrics['test_r2']:.3f}")

        return metrics

    def save(self, path: str):
        """Save model and artifacts"""

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca_dict': self.pca_dict,
            'feature_names': self.feature_names,
            'y_99': self.y_99,
            'random_state': self.random_state
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        print(f"  Model saved: {path}")

    def load(self, path: str):
        """Load model and artifacts"""

        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca_dict = model_data['pca_dict']
        self.feature_names = model_data['feature_names']
        self.y_99 = model_data['y_99']
        self.random_state = model_data['random_state']

        print(f"  Model loaded: {path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""

        if self.model is None:
            raise ValueError("Model not trained or loaded!")

        # Ensure same columns as training
        X = X[self.feature_names]

        # Dummy y for preprocessing
        y_dummy = np.zeros(len(X))

        # Preprocess
        X_proc, _ = self.preprocess(X, y_dummy, fit=False)

        # Predict (log space)
        y_pred_log = self.model.predict(X_proc)

        # Transform back
        y_pred = np.expm1(y_pred_log)

        return y_pred


def run_experiment(config: ExperimentConfig,
                  X: pd.DataFrame,
                  y: np.ndarray,
                  save_model: bool = True) -> ExperimentResult:
    """
    Run complete experiment with tracking

    Args:
        config: Experiment configuration
        X: Feature matrix
        y: Target variable
        save_model: Whether to save trained model

    Returns:
        ExperimentResult
    """

    tracker = get_tracker()

    # Start experiment
    experiment_id = tracker.start_experiment(config)
    start_time = time.time()

    # Train
    trainer = ModelTrainer(random_state=config.random_state)

    metrics = trainer.train(
        X=X,
        y=y,
        model_type=config.model_type,
        model_params=config.model_params,
        pca_components=config.preprocessing.get('pca_components'),
        test_size=config.preprocessing.get('test_size', 0.2)
    )

    # Save model
    model_path = None
    if save_model:
        model_path = f"models/{experiment_id}.pkl"
        trainer.save(model_path)

    # Create result
    duration = time.time() - start_time

    result = ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        metrics=metrics,
        timestamp=datetime.now().isoformat(),
        duration_seconds=duration,
        model_path=model_path
    )

    # Log result
    tracker.log_result(result)

    return result
