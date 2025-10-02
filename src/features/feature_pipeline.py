"""Feature engineering pipeline."""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from src.utils import load_config, get_data_path, setup_logger
from .baseline_features import BaselineFeatureExtractor


class FeaturePipeline:
    """Pipeline for feature engineering and dataset preparation."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize feature pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            'FeaturePipeline',
            log_file=self.config['logging']['file'],
            level=self.config['logging']['level']
        )

        # Initialize feature extractor
        self.extractor = BaselineFeatureExtractor(
            follower_count=self.config['instagram']['follower_count']
        )

        self.logger.info("FeaturePipeline initialized")

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw Instagram data.

        Returns:
            Raw dataframe

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        data_path = get_data_path(
            self.config['data']['raw_csv'],
            data_type='raw'
        )

        self.logger.info(f"Loading raw data from {data_path}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(df)} posts")

        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from raw data.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with extracted features
        """
        self.logger.info("Starting feature extraction...")

        features_df = self.extractor.transform(df)

        self.logger.info(f"Feature extraction complete: {features_df.shape}")

        return features_df

    def create_stratified_bins(self, y: pd.Series, n_bins: int = 3) -> pd.Series:
        """Create stratification bins from continuous target.

        Args:
            y: Target variable (likes)
            n_bins: Number of bins for stratification

        Returns:
            Series with bin labels
        """
        bins = pd.qcut(y, q=n_bins, labels=['low', 'medium', 'high'], duplicates='drop')
        return bins

    def split_data(
        self,
        features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets with stratification.

        Args:
            features_df: DataFrame with features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Splitting data into train/test sets...")

        # Get feature names
        feature_cols = self.extractor.get_feature_names()

        # Prepare X and y
        X = features_df[feature_cols].copy()
        y = features_df['likes'].copy()

        # Create stratification bins
        if self.config['training']['stratify']:
            strat_bins = self.create_stratified_bins(
                y,
                n_bins=self.config['training']['stratify_bins']
            )
            stratify = strat_bins
            self.logger.info(f"Using stratified split with {self.config['training']['stratify_bins']} bins")
        else:
            stratify = None
            self.logger.info("Using random split (no stratification)")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=stratify
        )

        self.logger.info(f"Train set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        self.logger.info(f"Train likes - mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
        self.logger.info(f"Test likes - mean: {y_test.mean():.2f}, std: {y_test.std():.2f}")

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, features_df: pd.DataFrame):
        """Save processed features to CSV.

        Args:
            features_df: DataFrame with features
        """
        output_path = get_data_path(
            Path(self.config['data']['processed']).name,
            data_type='processed'
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        features_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed data to {output_path}")

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Run the complete feature engineering pipeline.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("=" * 70)
        self.logger.info("Starting Feature Engineering Pipeline")
        self.logger.info("=" * 70)

        # Load raw data
        df_raw = self.load_raw_data()

        # Extract features
        features_df = self.extract_features(df_raw)

        # Save processed data
        self.save_processed_data(features_df)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(features_df)

        self.logger.info("=" * 70)
        self.logger.info("Feature Engineering Pipeline Complete")
        self.logger.info("=" * 70)

        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Run pipeline
    pipeline = FeaturePipeline()
    X_train, X_test, y_train, y_test = pipeline.run()

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nFeature names:")
    for i, feat in enumerate(X_train.columns, 1):
        print(f"  {i}. {feat}")
    print("\n" + "=" * 70)
