"""
Visual Feature Extractor for Instagram Posts
=============================================

Based on research findings:
- Google Cloud Vision API achieves 6.8% higher accuracy
- Face detection boosts engagement prediction
- Color analysis important for engagement
- Image quality metrics significant

Features extracted:
1. Face detection (OpenCV Haar Cascade)
2. Color analysis (dominant colors, palette)
3. Brightness & contrast
4. Image sharpness/blur
5. Aspect ratio & composition
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class VisualFeatureExtractor:
    """Extract visual features from Instagram images."""

    def __init__(self):
        """Initialize with Haar Cascade for face detection."""
        # Load pre-trained Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.feature_names = [
            'face_count',              # Number of faces detected
            'has_face',                # Binary: contains face
            'brightness',              # Average brightness (0-255)
            'contrast',                # Image contrast score
            'saturation',              # Color saturation (HSV)
            'dominant_hue',            # Dominant color hue (0-180)
            'color_diversity',         # Number of distinct colors
            'sharpness',               # Image sharpness (blur detection)
            'aspect_ratio',            # Width/height ratio
            'is_square',               # Binary: square image (1:1)
            'is_portrait',             # Binary: portrait orientation
            'is_landscape',            # Binary: landscape orientation
        ]

    def extract_from_image(self, image_path: str) -> Dict[str, float]:
        """Extract visual features from a single image."""
        features = {}

        try:
            # Load image
            img = cv2.imread(str(image_path))

            if img is None:
                # Return default values if image can't be loaded
                return self._default_features()

            # 1. FACE DETECTION (Research: significant for engagement)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            features['face_count'] = len(faces)
            features['has_face'] = 1 if len(faces) > 0 else 0

            # 2. BRIGHTNESS (Research: bright images get more engagement)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])  # V channel
            features['brightness'] = brightness

            # 3. CONTRAST (Research: high contrast = more engaging)
            contrast = np.std(gray)
            features['contrast'] = contrast

            # 4. SATURATION (Research: colorful images perform better)
            saturation = np.mean(hsv[:, :, 1])  # S channel
            features['saturation'] = saturation

            # 5. DOMINANT COLOR (Research: certain colors boost engagement)
            # Get dominant hue (color)
            hue_channel = hsv[:, :, 0]
            dominant_hue = np.median(hue_channel)
            features['dominant_hue'] = dominant_hue

            # 6. COLOR DIVERSITY (Research: varied colors = interesting)
            # Count distinct colors (simplified)
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            # Normalize by image size
            color_diversity = unique_colors / (img.shape[0] * img.shape[1]) * 1000
            features['color_diversity'] = min(color_diversity, 100)  # Cap at 100

            # 7. SHARPNESS (Research: sharp images > blurry)
            # Laplacian variance (blur detection)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            features['sharpness'] = sharpness

            # 8. ASPECT RATIO & COMPOSITION
            height, width = img.shape[:2]
            aspect_ratio = width / height if height > 0 else 1.0
            features['aspect_ratio'] = aspect_ratio

            # Instagram formats
            features['is_square'] = 1 if 0.95 <= aspect_ratio <= 1.05 else 0
            features['is_portrait'] = 1 if aspect_ratio < 0.95 else 0
            features['is_landscape'] = 1 if aspect_ratio > 1.05 else 0

        except Exception as e:
            # If any error, return default features
            print(f"Warning: Could not process {image_path}: {e}")
            return self._default_features()

        return features

    def _default_features(self) -> Dict[str, float]:
        """Return default feature values when image processing fails."""
        return {
            'face_count': 0,
            'has_face': 0,
            'brightness': 128,  # Middle brightness
            'contrast': 50,
            'saturation': 50,
            'dominant_hue': 90,  # Green (middle of spectrum)
            'color_diversity': 50,
            'sharpness': 100,
            'aspect_ratio': 1.0,
            'is_square': 1,
            'is_portrait': 0,
            'is_landscape': 0,
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract visual features from all posts.

        Args:
            df: DataFrame with 'file_path' column

        Returns:
            DataFrame with visual features
        """
        print(f"Extracting visual features from {len(df)} posts...")

        all_features = []

        for idx, row in df.iterrows():
            file_path = row.get('file_path', '')

            # Check if it's a video (skip visual features for videos)
            is_video = row.get('is_video', 0)

            if is_video:
                # Videos: use default features
                features = self._default_features()
            else:
                # Images: extract features
                features = self.extract_from_image(file_path)

            all_features.append(features)

            # Print progress every 20 posts (more frequent)
            if (idx + 1) % 20 == 0:
                progress = (idx + 1) / len(df) * 100
                print(f"  Progress: {idx + 1}/{len(df)} ({progress:.1f}%)")

        features_df = pd.DataFrame(all_features)

        print(f"✅ Extracted {len(self.feature_names)} visual features")
        print(f"✅ Feature names: {self.feature_names}")

        # Add some statistics
        print(f"\n📊 Visual Feature Statistics:")
        print(f"   Posts with faces: {features_df['has_face'].sum()} ({features_df['has_face'].sum()/len(features_df)*100:.1f}%)")
        print(f"   Avg face count: {features_df['face_count'].mean():.2f}")
        print(f"   Avg brightness: {features_df['brightness'].mean():.1f}")
        print(f"   Avg sharpness: {features_df['sharpness'].mean():.1f}")
        print(f"   Square images: {features_df['is_square'].sum()} ({features_df['is_square'].sum()/len(features_df)*100:.1f}%)")

        return features_df


class AdvancedVisualFeatureExtractor(VisualFeatureExtractor):
    """
    Extended visual features with more advanced analysis.

    Additional features:
    - Edge density (composition complexity)
    - Texture analysis
    - Color histogram features
    """

    def __init__(self):
        super().__init__()

        # Add additional features
        self.feature_names.extend([
            'edge_density',         # Amount of edges (composition complexity)
            'warm_color_ratio',     # Ratio of warm colors (red/orange/yellow)
            'cool_color_ratio',     # Ratio of cool colors (blue/green)
            'high_brightness_ratio',# Ratio of very bright pixels
            'low_brightness_ratio', # Ratio of dark pixels
        ])

    def extract_from_image(self, image_path: str) -> Dict[str, float]:
        """Extract extended visual features."""
        # Get base features
        features = super().extract_from_image(image_path)

        try:
            # Load image again for advanced features
            img = cv2.imread(str(image_path))

            if img is None:
                features.update(self._default_advanced_features())
                return features

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 1. EDGE DENSITY (composition complexity)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features['edge_density'] = edge_density * 100  # Scale to 0-100

            # 2. COLOR TEMPERATURE
            hue = hsv[:, :, 0]
            # Warm colors: red (0-30, 150-180), orange (30-45), yellow (45-75)
            warm_mask = ((hue >= 0) & (hue <= 45)) | ((hue >= 150) & (hue <= 180))
            warm_ratio = np.sum(warm_mask) / (hue.shape[0] * hue.shape[1])
            features['warm_color_ratio'] = warm_ratio

            # Cool colors: green (75-150), blue (150-180 handled above)
            cool_mask = (hue >= 75) & (hue <= 150)
            cool_ratio = np.sum(cool_mask) / (hue.shape[0] * hue.shape[1])
            features['cool_color_ratio'] = cool_ratio

            # 3. BRIGHTNESS DISTRIBUTION
            brightness = hsv[:, :, 2]
            high_bright = np.sum(brightness > 200) / (brightness.shape[0] * brightness.shape[1])
            low_bright = np.sum(brightness < 50) / (brightness.shape[0] * brightness.shape[1])
            features['high_brightness_ratio'] = high_bright
            features['low_brightness_ratio'] = low_bright

        except Exception as e:
            features.update(self._default_advanced_features())

        return features

    def _default_advanced_features(self) -> Dict[str, float]:
        """Default values for advanced features."""
        return {
            'edge_density': 10,
            'warm_color_ratio': 0.3,
            'cool_color_ratio': 0.3,
            'high_brightness_ratio': 0.2,
            'low_brightness_ratio': 0.1,
        }

    def _default_features(self) -> Dict[str, float]:
        """Override to include advanced features."""
        base = super()._default_features()
        base.update(self._default_advanced_features())
        return base


if __name__ == '__main__':
    """Test visual feature extraction."""
    import sys

    # Load data
    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    print(f"Loaded {len(df)} posts")

    # Test basic extractor
    print("\n" + "="*80)
    print("TESTING BASIC VISUAL FEATURE EXTRACTOR")
    print("="*80)

    extractor = VisualFeatureExtractor()
    features_df = extractor.transform(df.head(10))  # Test on first 10
    print("\nSample features:")
    print(features_df.head())

    # Test advanced extractor
    print("\n" + "="*80)
    print("TESTING ADVANCED VISUAL FEATURE EXTRACTOR")
    print("="*80)

    adv_extractor = AdvancedVisualFeatureExtractor()
    adv_features_df = adv_extractor.transform(df.head(10))  # Test on first 10
    print("\nSample advanced features:")
    print(adv_features_df.head())
