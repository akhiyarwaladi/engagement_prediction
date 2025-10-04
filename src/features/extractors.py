#!/usr/bin/env python3
"""
Modular Feature Extractors
All feature extraction logic centralized and reusable
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import cv2
from PIL import Image
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors"""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from dataframe"""
        pass

    def fit(self, df: pd.DataFrame):
        """Fit any trainable components"""
        self.is_fitted = True
        return self

    def fit_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and extract in one step"""
        self.fit(df)
        return self.extract(df)


class BaselineFeatureExtractor(BaseFeatureExtractor):
    """Extract baseline hand-crafted features"""

    def __init__(self):
        super().__init__("baseline")

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract baseline features"""
        df = df.copy()

        # Caption features
        df['caption'] = df['caption'].fillna('')
        df['caption_length'] = df['caption'].str.len()
        df['word_count'] = df['caption'].str.split().str.len()
        df['hashtag_count'] = df['hashtags_count']
        df['mention_count'] = df['mentions_count']

        # Media type
        df['is_video'] = df['is_video'].astype(int)

        # Temporal features
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['date'].dt.month

        features = [
            'caption_length', 'word_count', 'hashtag_count', 'mention_count',
            'is_video', 'is_weekend'
        ]

        return df[features]


class CyclicTemporalExtractor(BaseFeatureExtractor):
    """Extract cyclic temporal features (sine/cosine encoding)"""

    def __init__(self):
        super().__init__("cyclic_temporal")

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cyclic features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Hour (24-hour cycle)
        hour = df['date'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Day of week (7-day cycle)
        day = df['date'].dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * day / 7)
        df['day_cos'] = np.cos(2 * np.pi * day / 7)

        # Month (12-month cycle)
        month = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

        features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos'
        ]

        return df[features]


class AdvancedTemporalExtractor(BaseFeatureExtractor):
    """Extract advanced temporal features (momentum, frequency, etc.)"""

    def __init__(self):
        super().__init__("advanced_temporal")

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced temporal features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Days since last post
        df['days_since_last_post'] = df['date'].diff().dt.total_seconds() / 86400
        df['days_since_last_post'].fillna(df['days_since_last_post'].median(), inplace=True)

        # Posting frequency (posts per week)
        df['posting_frequency'] = df.groupby(pd.Grouper(key='date', freq='7D')).size().reindex(
            df['date']).fillna(method='ffill').fillna(1).values

        # Trend momentum
        df['likes_ma5'] = df['likes'].rolling(window=5, min_periods=1).mean()
        df['trend_momentum'] = (df['likes'] - df['likes_ma5']) / (df['likes_ma5'] + 1)

        # Days since first post
        df['days_since_first_post'] = (df['date'] - df['date'].min()).dt.total_seconds() / 86400

        # Engagement velocity
        df['engagement_velocity'] = df['likes'] / (df['days_since_last_post'] + 1)

        features = [
            'days_since_last_post', 'posting_frequency', 'trend_momentum',
            'days_since_first_post', 'engagement_velocity'
        ]

        return df[features]


class EngagementLagExtractor(BaseFeatureExtractor):
    """Extract engagement lag features (historical performance)"""

    def __init__(self, lags: List[int] = [1, 2, 3, 5]):
        super().__init__("engagement_lag")
        self.lags = lags

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract lag features"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        features = []

        # Lag features
        for lag in self.lags:
            col_name = f'likes_lag_{lag}'
            df[col_name] = df['likes'].shift(lag)
            df[col_name].fillna(df['likes'].median(), inplace=True)
            features.append(col_name)

        # Rolling statistics
        df['likes_rolling_mean_5'] = df['likes'].rolling(window=5, min_periods=1).mean()
        df['likes_rolling_std_5'] = df['likes'].rolling(window=5, min_periods=1).std().fillna(0)
        features.extend(['likes_rolling_mean_5', 'likes_rolling_std_5'])

        return df[features]


class BERTEmbeddingExtractor(BaseFeatureExtractor):
    """Extract IndoBERT text embeddings"""

    def __init__(self, model_name: str = "indobenchmark/indobert-base-p1",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 cache_path: Optional[str] = 'data/processed/bert_embeddings.csv'):
        super().__init__("bert")
        self.model_name = model_name
        self.device = device
        self.cache_path = cache_path
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Load BERT model"""
        if self.tokenizer is None:
            print(f"  Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract BERT embeddings"""

        # Try loading from cache
        if self.cache_path and Path(self.cache_path).exists():
            print(f"  Loading cached BERT embeddings from {self.cache_path}")
            return pd.read_csv(self.cache_path)

        # Extract fresh
        self._load_model()
        print(f"  Extracting BERT embeddings for {len(df)} posts...")

        embeddings_list = []

        for idx, row in df.iterrows():
            caption = str(row['caption']) if pd.notna(row['caption']) else ""

            # Tokenize and encode
            inputs = self.tokenizer(caption, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

            embeddings_list.append(embedding)

            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{len(df)}")

        # Create DataFrame
        embeddings_df = pd.DataFrame(
            embeddings_list,
            columns=[f'bert_{i}' for i in range(768)]
        )
        embeddings_df['post_id'] = df['post_id'].values

        # Cache
        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            embeddings_df.to_csv(self.cache_path, index=False)
            print(f"  Cached to {self.cache_path}")

        # Cleanup
        del self.model, self.tokenizer
        torch.cuda.empty_cache()

        return embeddings_df


class ViTEmbeddingExtractor(BaseFeatureExtractor):
    """Extract Vision Transformer embeddings (with video support)"""

    def __init__(self, model_name: str = "google/vit-base-patch16-224",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 n_frames: int = 3,
                 cache_path: Optional[str] = 'data/processed/vit_embeddings_enhanced.csv'):
        super().__init__("vit")
        self.model_name = model_name
        self.device = device
        self.n_frames = n_frames
        self.cache_path = cache_path
        self.processor = None
        self.model = None

    def _load_model(self):
        """Load ViT model"""
        if self.processor is None:
            print(f"  Loading {self.model_name}...")
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def _extract_video_frames(self, video_path: Path) -> Optional[List[Image.Image]]:
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return None

            frame_indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))

            cap.release()
            return frames if frames else None
        except Exception:
            return None

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract ViT embeddings"""

        # Try loading from cache
        if self.cache_path and Path(self.cache_path).exists():
            print(f"  Loading cached ViT embeddings from {self.cache_path}")
            return pd.read_csv(self.cache_path)

        # Extract fresh
        self._load_model()
        print(f"  Extracting ViT embeddings for {len(df)} posts...")

        embeddings_list = []

        for idx, row in df.iterrows():
            file_path = Path(row['file_path'])

            if not file_path.exists():
                embedding = np.zeros(768)
            elif row['is_video']:
                # Extract frames from video
                frames = self._extract_video_frames(file_path)
                if frames:
                    frame_embeddings = []
                    for frame in frames:
                        inputs = self.processor(images=frame, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                            frame_embeddings.append(emb)

                    embedding = np.mean(frame_embeddings, axis=0)
                else:
                    embedding = np.zeros(768)
            else:
                # Process image
                image = Image.open(file_path).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

            embeddings_list.append(embedding)

            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{len(df)}")

        # Create DataFrame
        embeddings_df = pd.DataFrame(
            embeddings_list,
            columns=[f'vit_{i}' for i in range(768)]
        )
        embeddings_df['post_id'] = df['post_id'].values

        # Cache
        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            embeddings_df.to_csv(self.cache_path, index=False)
            print(f"  Cached to {self.cache_path}")

        # Cleanup
        del self.model, self.processor
        torch.cuda.empty_cache()

        return embeddings_df


class FeaturePipeline:
    """Orchestrate multiple feature extractors"""

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        self.extractors = extractors

    def extract_all(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract all features and combine"""

        print(f"\n[PIPELINE] Extracting features with {len(self.extractors)} extractors...")

        df_processed = df.copy()
        all_features = []

        for extractor in self.extractors:
            print(f"\n[{extractor.name.upper()}] Extracting...")
            features_df = extractor.extract(df_processed)
            all_features.append(features_df)
            print(f"  Extracted {features_df.shape[1]} features")

        # Combine all features
        X = pd.concat(all_features, axis=1)

        print(f"\n[PIPELINE] Total features extracted: {X.shape[1]}")

        return X, df_processed

    def add_extractor(self, extractor: BaseFeatureExtractor):
        """Add an extractor to the pipeline"""
        self.extractors.append(extractor)
        return self


# Convenience function to create standard pipeline
def create_standard_pipeline(
    use_bert: bool = True,
    use_vit: bool = True,
    use_advanced_temporal: bool = True,
    use_lag_features: bool = True
) -> FeaturePipeline:
    """Create standard feature extraction pipeline"""

    extractors = [
        BaselineFeatureExtractor(),
        CyclicTemporalExtractor(),
    ]

    if use_advanced_temporal:
        extractors.append(AdvancedTemporalExtractor())

    if use_lag_features:
        extractors.append(EngagementLagExtractor())

    if use_bert:
        extractors.append(BERTEmbeddingExtractor())

    if use_vit:
        extractors.append(ViTEmbeddingExtractor())

    return FeaturePipeline(extractors)
