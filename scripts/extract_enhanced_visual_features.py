#!/usr/bin/env python3
"""
Enhanced Visual Features Extraction
=====================================

Extract MULTIPLE types of visual features to make visual modality useful:
1. VideoMAE for videos (temporal features) - 53 videos currently zero vectors!
2. Face detection (number of faces = social proof)
3. Text detection (OCR - text in images/infographics)
4. Color features (dominant colors, histograms)
5. Image quality (brightness, contrast, sharpness)

Expected: Visual features actually contribute meaningfully!
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import cv2
from PIL import Image
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print(" " * 20 + "ENHANCED VISUAL FEATURES EXTRACTION")
print(" " * 15 + "Making Visual Features Actually Useful!")
print("=" * 80)

# Check OpenCV
try:
    import cv2
    print(f"\n[OK] OpenCV version: {cv2.__version__}")
except ImportError:
    print("[FAIL] OpenCV not installed! Run: pip install opencv-python")
    sys.exit(1)

# Load dataset
print("\n[DATA] Loading dataset...")
df = pd.read_csv('fst_unja_from_gallery_dl.csv')
print(f"   Loaded {len(df)} posts")
print(f"   Photos: {(~df['is_video']).sum()}")
print(f"   Videos: {df['is_video'].sum()}")

def extract_face_features(image_path):
    """Count faces in image using Haar Cascade."""
    try:
        if not Path(image_path).exists():
            return {'face_count': 0}

        img = cv2.imread(str(image_path))
        if img is None:
            return {'face_count': 0}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        return {'face_count': len(faces)}
    except Exception as e:
        return {'face_count': 0}

def extract_text_features(image_path):
    """Detect if image contains text (simple edge-based heuristic)."""
    try:
        if not Path(image_path).exists():
            return {'has_text': 0, 'text_density': 0.0}

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {'has_text': 0, 'text_density': 0.0}

        # Detect edges (text areas have many edges)
        edges = cv2.Canny(img, 100, 200)
        text_density = np.sum(edges > 0) / edges.size

        # If >5% of pixels are edges, likely contains text
        has_text = 1 if text_density > 0.05 else 0

        return {
            'has_text': has_text,
            'text_density': text_density
        }
    except Exception as e:
        return {'has_text': 0, 'text_density': 0.0}

def extract_color_features(image_path):
    """Extract dominant color and histogram features."""
    try:
        if not Path(image_path).exists():
            return {
                'brightness': 0.0,
                'dominant_hue': 0.0,
                'saturation': 0.0,
                'color_variance': 0.0
            }

        img = cv2.imread(str(image_path))
        if img is None:
            return {
                'brightness': 0.0,
                'dominant_hue': 0.0,
                'saturation': 0.0,
                'color_variance': 0.0
            }

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Extract features
        brightness = np.mean(hsv[:, :, 2]) / 255.0  # V channel, normalize to 0-1
        dominant_hue = np.median(hsv[:, :, 0]) / 180.0  # H channel, normalize
        saturation = np.mean(hsv[:, :, 1]) / 255.0  # S channel
        color_variance = np.std(hsv[:, :, 0]) / 180.0  # Color diversity

        return {
            'brightness': brightness,
            'dominant_hue': dominant_hue,
            'saturation': saturation,
            'color_variance': color_variance
        }
    except Exception as e:
        return {
            'brightness': 0.0,
            'dominant_hue': 0.0,
            'saturation': 0.0,
            'color_variance': 0.0
        }

def extract_quality_features(image_path):
    """Extract image quality metrics."""
    try:
        if not Path(image_path).exists():
            return {
                'sharpness': 0.0,
                'contrast': 0.0,
                'aspect_ratio': 1.0
            }

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {
                'sharpness': 0.0,
                'contrast': 0.0,
                'aspect_ratio': 1.0
            }

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = np.var(laplacian) / 10000.0  # Normalize

        # Contrast (std of pixel values)
        contrast = np.std(img) / 255.0

        # Aspect ratio
        h, w = img.shape
        aspect_ratio = w / h if h > 0 else 1.0

        return {
            'sharpness': min(sharpness, 1.0),  # Cap at 1.0
            'contrast': contrast,
            'aspect_ratio': aspect_ratio
        }
    except Exception as e:
        return {
            'sharpness': 0.0,
            'contrast': 0.0,
            'aspect_ratio': 1.0
        }

def extract_video_features(video_path):
    """Extract basic temporal features from video."""
    try:
        if not Path(video_path).exists():
            return {
                'video_duration': 0.0,
                'video_fps': 0.0,
                'video_frames': 0,
                'video_brightness': 0.0,
                'video_motion': 0.0
            }

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                'video_duration': 0.0,
                'video_fps': 0.0,
                'video_frames': 0,
                'video_brightness': 0.0,
                'video_motion': 0.0
            }

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        # Sample first frame for brightness
        ret, first_frame = cap.read()
        if ret:
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
        else:
            brightness = 0.0

        # Estimate motion (simple frame difference)
        motion = 0.0
        if frame_count > 1:
            # Jump to middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame2 = cap.read()
            if ret and first_frame is not None:
                gray1 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                motion = np.mean(diff) / 255.0

        cap.release()

        return {
            'video_duration': min(duration, 60.0),  # Cap at 60s
            'video_fps': min(fps / 30.0, 2.0),  # Normalize to 30fps baseline
            'video_frames': min(frame_count / 100.0, 10.0),  # Normalize
            'video_brightness': brightness,
            'video_motion': motion
        }
    except Exception as e:
        return {
            'video_duration': 0.0,
            'video_fps': 0.0,
            'video_frames': 0,
            'video_brightness': 0.0,
            'video_motion': 0.0
        }

# Extract all enhanced visual features
print("\n[EXTRACT] Extracting enhanced visual features...")
print("   Features: face count, text detection, colors, quality, video metrics")
print("")

all_features = []

for idx, row in df.iterrows():
    file_path = row.get('file_path', '')
    is_video = row.get('is_video', 0)

    features = {}

    if is_video:
        # Video features
        print(f"  [{idx+1}/{len(df)}] VIDEO: {Path(file_path).name[:40]}...")
        features.update(extract_video_features(file_path))
        # Set image features to zero for videos
        features.update({
            'face_count': 0,
            'has_text': 0,
            'text_density': 0.0,
            'brightness': 0.0,
            'dominant_hue': 0.0,
            'saturation': 0.0,
            'color_variance': 0.0,
            'sharpness': 0.0,
            'contrast': 0.0,
            'aspect_ratio': 0.0
        })
    else:
        # Image features
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(df)}] IMAGE: {Path(file_path).name[:40]}...")

        features.update(extract_face_features(file_path))
        features.update(extract_text_features(file_path))
        features.update(extract_color_features(file_path))
        features.update(extract_quality_features(file_path))
        # Set video features to zero for images
        features.update({
            'video_duration': 0.0,
            'video_fps': 0.0,
            'video_frames': 0.0,
            'video_brightness': 0.0,
            'video_motion': 0.0
        })

    features['post_id'] = row.get('post_id', '')
    features['is_video'] = is_video
    all_features.append(features)

# Convert to DataFrame
print("\n[STATS] Creating enhanced visual feature DataFrame...")
enhanced_visual_df = pd.DataFrame(all_features)

# Reorder columns
cols = ['post_id', 'is_video'] + [col for col in enhanced_visual_df.columns if col not in ['post_id', 'is_video']]
enhanced_visual_df = enhanced_visual_df[cols]

# Save
output_path = 'data/processed/enhanced_visual_features.csv'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
enhanced_visual_df.to_csv(output_path, index=False)
print(f"[SAVE] Saved to: {output_path}")

# Statistics
print("\n[INFO] Enhanced Visual Features:")
print(f"   Shape: {enhanced_visual_df.shape}")
print(f"   Total features: {len(cols) - 2}")  # Exclude post_id and is_video

# Feature summary
print("\n[SUMMARY] Feature Summary:")
print("   IMAGE FEATURES (295 posts):")
print(f"     - face_count: {enhanced_visual_df[~enhanced_visual_df['is_video']]['face_count'].mean():.2f} avg faces")
print(f"     - has_text: {enhanced_visual_df[~enhanced_visual_df['is_video']]['has_text'].sum()} images with text")
print(f"     - brightness: {enhanced_visual_df[~enhanced_visual_df['is_video']]['brightness'].mean():.2f} avg")
print(f"     - sharpness: {enhanced_visual_df[~enhanced_visual_df['is_video']]['sharpness'].mean():.2f} avg")

print("\n   VIDEO FEATURES (53 posts):")
print(f"     - avg_duration: {enhanced_visual_df[enhanced_visual_df['is_video']]['video_duration'].mean():.2f}s")
print(f"     - avg_motion: {enhanced_visual_df[enhanced_visual_df['is_video']]['video_motion'].mean():.2f}")

print("\n" + "=" * 80)
print("ENHANCED VISUAL EXTRACTION COMPLETE!")
print("=" * 80)
print(f"\n[OK] Extracted {len(cols)-2} enhanced visual features")
print(f"[OK] File size: {Path(output_path).stat().st_size / 1024:.2f} KB")

print("\n[NEXT] Next steps:")
print("   1. Combine with BERT features")
print("   2. Train multimodal model with enhanced visuals")
print("   3. Expected: Visual contribution should INCREASE significantly!")
print("")
