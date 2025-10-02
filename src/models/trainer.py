"""Model training orchestration."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_config, get_model_path, setup_logger, get_project_root
from src.features import FeaturePipeline
from .baseline_model import BaselineModel


class ModelTrainer:
    """Orchestrate model training, evaluation, and persistence."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize trainer.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            'ModelTrainer',
            log_file=self.config['logging']['file']
        )

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.logger.info("ModelTrainer initialized")

    def load_data(self):
        """Load training and test data from feature pipeline."""
        self.logger.info("Loading data via feature pipeline...")

        pipeline = FeaturePipeline()
        self.X_train, self.X_test, self.y_train, self.y_test = pipeline.run()

        self.logger.info(f"Data loaded: Train={len(self.X_train)}, Test={len(self.X_test)}")

    def train(self):
        """Train baseline Random Forest model."""
        self.logger.info("=" * 70)
        self.logger.info("Training Baseline Model (Random Forest)")
        self.logger.info("=" * 70)

        # Get model parameters from config
        model_params = self.config['model']['random_forest']

        # Initialize and train model
        self.model = BaselineModel(**model_params)

        self.logger.info(f"Model parameters: {model_params}")
        self.logger.info("Training model...")

        self.model.fit(self.X_train, self.y_train)

        self.logger.info("✅ Model training complete")

    def cross_validate(self, cv: int = 5) -> Tuple[float, float]:
        """Perform cross-validation.

        Args:
            cv: Number of cross-validation folds

        Returns:
            Tuple of (mean_score, std_score)
        """
        self.logger.info(f"Performing {cv}-fold cross-validation...")

        scores = cross_val_score(
            self.model.model,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        # Convert to positive MAE
        scores = -scores

        mean_score = scores.mean()
        std_score = scores.std()

        self.logger.info(f"Cross-validation MAE: {mean_score:.2f} ± {std_score:.2f}")

        return mean_score, std_score

    def evaluate(self) -> pd.DataFrame:
        """Evaluate model on test set.

        Returns:
            DataFrame with evaluation metrics
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score
        )

        self.logger.info("Evaluating model on test set...")

        # Make predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # Calculate metrics
        metrics = {
            'MAE_train': mean_absolute_error(self.y_train, y_pred_train),
            'MAE_test': mean_absolute_error(self.y_test, y_pred_test),
            'RMSE_train': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'RMSE_test': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'R2_train': r2_score(self.y_train, y_pred_train),
            'R2_test': r2_score(self.y_test, y_pred_test)
        }

        # MAPE (avoid division by zero)
        mask_train = self.y_train != 0
        mask_test = self.y_test != 0

        mape_train = np.mean(
            np.abs((self.y_train[mask_train] - y_pred_train[mask_train]) / self.y_train[mask_train])
        ) * 100

        mape_test = np.mean(
            np.abs((self.y_test[mask_test] - y_pred_test[mask_test]) / self.y_test[mask_test])
        ) * 100

        metrics['MAPE_train'] = mape_train
        metrics['MAPE_test'] = mape_test

        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics])

        # Log metrics
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EVALUATION METRICS")
        self.logger.info("=" * 70)
        for metric, value in metrics.items():
            self.logger.info(f"{metric:15s}: {value:.4f}")
        self.logger.info("=" * 70)

        # Check against target performance
        target_mae = self.config['evaluation']['target_performance']['mae_max']
        target_r2 = self.config['evaluation']['target_performance']['r2_min']

        if metrics['MAE_test'] <= target_mae:
            self.logger.info(f"✅ MAE target achieved: {metrics['MAE_test']:.2f} <= {target_mae}")
        else:
            self.logger.warning(f"⚠️ MAE target NOT met: {metrics['MAE_test']:.2f} > {target_mae}")

        if metrics['R2_test'] >= target_r2:
            self.logger.info(f"✅ R² target achieved: {metrics['R2_test']:.3f} >= {target_r2}")
        else:
            self.logger.warning(f"⚠️ R² target NOT met: {metrics['R2_test']:.3f} < {target_r2}")

        return metrics_df

    def plot_feature_importance(self, top_n: int = 10):
        """Plot feature importance.

        Args:
            top_n: Number of top features to plot
        """
        importance_df = self.model.get_feature_importance()
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        # Save plot
        output_dir = get_project_root() / self.config['output']['figures']
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Feature importance plot saved to {output_path}")

        plt.close()

        return importance_df

    def plot_predictions(self):
        """Plot actual vs predicted values."""
        y_pred_test = self.model.predict(self.X_test)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot
        axes[0].scatter(self.y_test, y_pred_test, alpha=0.6, edgecolors='k')
        axes[0].plot([self.y_test.min(), self.y_test.max()],
                     [self.y_test.min(), self.y_test.max()],
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Likes')
        axes[0].set_ylabel('Predicted Likes')
        axes[0].set_title('Actual vs Predicted Likes')
        axes[0].grid(True, alpha=0.3)

        # Residual plot
        residuals = self.y_test - y_pred_test
        axes[1].scatter(y_pred_test, residuals, alpha=0.6, edgecolors='k')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Likes')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_dir = get_project_root() / self.config['output']['figures']
        output_path = output_dir / 'predictions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Predictions plot saved to {output_path}")

        plt.close()

    def save_model(self, filename: str = 'baseline_rf_model.pkl'):
        """Save trained model.

        Args:
            filename: Name of model file
        """
        model_path = get_model_path(filename)
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")

    def run(self) -> pd.DataFrame:
        """Run complete training pipeline.

        Returns:
            DataFrame with evaluation metrics
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STARTING MODEL TRAINING PIPELINE")
        self.logger.info("=" * 70)

        # Load data
        self.load_data()

        # Train model
        self.train()

        # Cross-validation
        cv_mae, cv_std = self.cross_validate()

        # Evaluate
        metrics_df = self.evaluate()

        # Feature importance
        importance_df = self.plot_feature_importance()

        # Plot predictions
        self.plot_predictions()

        # Save model
        self.save_model()

        self.logger.info("\n" + "=" * 70)
        self.logger.info("MODEL TRAINING PIPELINE COMPLETE")
        self.logger.info("=" * 70)

        return metrics_df


if __name__ == '__main__':
    # Run training
    trainer = ModelTrainer()
    metrics = trainer.run()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nFinal Metrics:")
    print(metrics.to_string())
    print("\n" + "=" * 70)
