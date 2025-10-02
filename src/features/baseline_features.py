"""Baseline feature extraction for Instagram posts.

This module extracts 9 simple features that don't require
complex NLP or computer vision processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime


class BaselineFeatureExtractor:
    """Extract baseline features from Instagram post data.

    Features extracted:
    1. caption_length: Number of characters in caption
    2. word_count: Number of words in caption
    3. hashtag_count: Number of hashtags used
    4. mention_count: Number of mentions (@username)
    5. is_video: Binary flag for video content
    6. hour: Hour of posting (0-23)
    7. day_of_week: Day of week (0=Monday, 6=Sunday)
    8. is_weekend: Binary flag for weekend posting
    9. month: Month of posting (1-12)
    """

    def __init__(self, follower_count: int = 4631):
        """Initialize feature extractor.

        Args:
            follower_count: Number of followers for engagement rate calculation
        """
        self.follower_count = follower_count
        self.feature_names = [
            'caption_length',
            'word_count',
            'hashtag_count',
            'mention_count',
            'is_video',
            'hour',
            'day_of_week',
            'is_weekend',
            'month'
        ]

    def extract_text_features(self, caption: str) -> Dict[str, Any]:
        """Extract features from caption text.

        Args:
            caption: Post caption text

        Returns:
            Dictionary with text features
        """
        if pd.isna(caption) or caption is None:
            caption = ""

        caption = str(caption)

        return {
            'caption_length': len(caption),
            'word_count': len(caption.split())
        }

    def extract_hashtag_features(self, hashtags: str, hashtag_count: int) -> Dict[str, Any]:
        """Extract features from hashtags.

        Args:
            hashtags: Space-separated hashtags
            hashtag_count: Pre-computed hashtag count

        Returns:
            Dictionary with hashtag features
        """
        return {
            'hashtag_count': hashtag_count if not pd.isna(hashtag_count) else 0
        }

    def extract_mention_features(self, mentions: str, mention_count: int) -> Dict[str, Any]:
        """Extract features from mentions.

        Args:
            mentions: Space-separated mentions
            mention_count: Pre-computed mention count

        Returns:
            Dictionary with mention features
        """
        return {
            'mention_count': mention_count if not pd.isna(mention_count) else 0
        }

    def extract_media_features(self, is_video: bool) -> Dict[str, Any]:
        """Extract features from media type.

        Args:
            is_video: Boolean flag for video content

        Returns:
            Dictionary with media features
        """
        return {
            'is_video': 1 if is_video else 0
        }

    def extract_temporal_features(self, date_str: str) -> Dict[str, Any]:
        """Extract features from posting date/time.

        Args:
            date_str: ISO format datetime string

        Returns:
            Dictionary with temporal features
        """
        try:
            dt = pd.to_datetime(date_str)
        except:
            # Default to noon on Monday if parsing fails
            dt = pd.Timestamp('2023-01-02 12:00:00')

        day_of_week = dt.dayofweek  # 0=Monday, 6=Sunday

        return {
            'hour': dt.hour,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'month': dt.month
        }

    def extract_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract all baseline features from a single post.

        Args:
            row: DataFrame row with post data

        Returns:
            Dictionary with all features
        """
        features = {}

        # Text features
        features.update(self.extract_text_features(row.get('caption', '')))

        # Hashtag features
        features.update(self.extract_hashtag_features(
            row.get('hashtags', ''),
            row.get('hashtags_count', 0)
        ))

        # Mention features
        features.update(self.extract_mention_features(
            row.get('mentions', ''),
            row.get('mentions_count', 0)
        ))

        # Media features
        features.update(self.extract_media_features(row.get('is_video', False)))

        # Temporal features
        features.update(self.extract_temporal_features(row.get('date', '')))

        return features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform entire dataframe by extracting features.

        Args:
            df: Input dataframe with raw Instagram data

        Returns:
            DataFrame with extracted features
        """
        print(f"Extracting baseline features from {len(df)} posts...")

        # Extract features for each row
        features_list = []
        for idx, row in df.iterrows():
            features = self.extract_features(row)
            features_list.append(features)

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        # Add target variable
        features_df['likes'] = df['likes'].values

        # Add engagement rate if follower count available
        features_df['engagement_rate'] = (features_df['likes'] / self.follower_count) * 100

        # Add metadata (useful for analysis)
        features_df['post_id'] = df['post_id'].values
        features_df['date'] = df['date'].values
        features_df['url'] = df['url'].values

        print(f"✅ Extracted {len(self.feature_names)} features")
        print(f"✅ Feature names: {self.feature_names}")

        return features_df

    def get_feature_names(self) -> list:
        """Get list of feature names.

        Returns:
            List of feature names
        """
        return self.feature_names.copy()
