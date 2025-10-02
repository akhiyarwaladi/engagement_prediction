#!/usr/bin/env python3
"""
Extract and cache visual features for all posts.
This separates feature extraction from model training for efficiency.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.features import AdvancedVisualFeatureExtractor

def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "VISUAL FEATURE EXTRACTION")
    print("=" * 80 + "\n")

    # Load data
    print("📁 Loading data...")
    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    print(f"   Loaded {len(df)} posts\n")

    # Extract visual features
    print("🎨 Extracting visual features from all images...")
    print("   (This may take 2-3 minutes for 271 posts)\n")

    extractor = AdvancedVisualFeatureExtractor()
    visual_features = extractor.transform(df)

    # Save to CSV
    output_path = 'data/processed/visual_features.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    visual_features.to_csv(output_path, index=False)
    print(f"\n💾 Visual features saved to: {output_path}")

    # Show sample
    print("\n📊 Sample of extracted features:")
    print(visual_features.head())

    print("\n" + "=" * 80)
    print("DONE! 🎉")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
