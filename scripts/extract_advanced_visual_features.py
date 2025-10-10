#!/usr/bin/env python3
"""
ADVANCED VISUAL FEATURES EXTRACTOR
Extract powerful visual features: Face detection, Color analysis, Image metadata
"""

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import json
from tqdm import tqdm

print("="*90)
print(" "*20 + "ADVANCED VISUAL FEATURES EXTRACTION")
print("="*90)
print()

# Load main dataset
df = pd.read_csv('multi_account_dataset.csv')
print(f"[LOAD] {len(df)} posts loaded")
print()

# Initialize face detector (Haar Cascade - fast and reliable)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

results = []

print("[EXTRACT] Processing images and videos...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    post_id = row['post_id']
    account = row['account']
    is_video = row['is_video']

    # Find image/video file
    if is_video:
        # For video, use thumbnail or first frame
        video_path = Path(f'gallery-dl/instagram/{account}') / f"{post_id}.mp4"
        if not video_path.exists():
            # Try alternative path
            video_path = Path(f'gallery-dl/instagram/{account}') / f"{post_id}.jpg"

        if video_path.exists() and video_path.suffix == '.mp4':
            # Extract first frame
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                # Skip if can't read video
                results.append({
                    'post_id': post_id,
                    'account': account,
                    'face_count': 0,
                    'has_faces': 0,
                    'dominant_r': 128, 'dominant_g': 128, 'dominant_b': 128,
                    'brightness': 128,
                    'saturation': 0,
                    'color_variance': 0,
                    'aspect_ratio': 1.0,
                    'resolution': 0,
                    'file_size_kb': 0,
                    'is_portrait': 0,
                    'is_landscape': 0,
                    'is_square': 1
                })
                continue
            img = frame
            img_path = video_path
        else:
            # Use thumbnail
            img_path = Path(f'gallery-dl/instagram/{account}') / f"{post_id}.jpg"
            if not img_path.exists():
                results.append({
                    'post_id': post_id,
                    'account': account,
                    'face_count': 0,
                    'has_faces': 0,
                    'dominant_r': 128, 'dominant_g': 128, 'dominant_b': 128,
                    'brightness': 128,
                    'saturation': 0,
                    'color_variance': 0,
                    'aspect_ratio': 1.0,
                    'resolution': 0,
                    'file_size_kb': 0,
                    'is_portrait': 0,
                    'is_landscape': 0,
                    'is_square': 1
                })
                continue
            img = cv2.imread(str(img_path))
    else:
        # Image
        img_path = Path(f'gallery-dl/instagram/{account}') / f"{post_id}.jpg"
        if not img_path.exists():
            # Try png
            img_path = Path(f'gallery-dl/instagram/{account}') / f"{post_id}.png"

        if not img_path.exists():
            results.append({
                'post_id': post_id,
                'account': account,
                'face_count': 0,
                'has_faces': 0,
                'dominant_r': 128, 'dominant_g': 128, 'dominant_b': 128,
                'brightness': 128,
                'saturation': 0,
                'color_variance': 0,
                'aspect_ratio': 1.0,
                'resolution': 0,
                'file_size_kb': 0,
                'is_portrait': 0,
                'is_landscape': 0,
                'is_square': 1
            })
            continue

        img = cv2.imread(str(img_path))

    if img is None:
        results.append({
            'post_id': post_id,
            'account': account,
            'face_count': 0,
            'has_faces': 0,
            'dominant_r': 128, 'dominant_g': 128, 'dominant_b': 128,
            'brightness': 128,
            'saturation': 0,
            'color_variance': 0,
            'aspect_ratio': 1.0,
            'resolution': 0,
            'file_size_kb': 0,
            'is_portrait': 0,
            'is_landscape': 0,
            'is_square': 1
        })
        continue

    # ========================================================================
    # 1. FACE DETECTION
    # ========================================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_count = len(faces)
    has_faces = 1 if face_count > 0 else 0

    # ========================================================================
    # 2. COLOR ANALYSIS
    # ========================================================================
    # Dominant color (mean RGB)
    dominant_b = int(np.mean(img[:, :, 0]))
    dominant_g = int(np.mean(img[:, :, 1]))
    dominant_r = int(np.mean(img[:, :, 2]))

    # Brightness (average intensity)
    brightness = int(np.mean(gray))

    # Saturation (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = int(np.mean(hsv[:, :, 1]))

    # Color variance (diversity)
    color_variance = float(np.std(img))

    # ========================================================================
    # 3. IMAGE METADATA
    # ========================================================================
    height, width = img.shape[:2]
    aspect_ratio = width / height if height > 0 else 1.0
    resolution = width * height
    file_size_kb = img_path.stat().st_size / 1024 if img_path.exists() else 0

    # Orientation
    is_portrait = 1 if aspect_ratio < 0.9 else 0
    is_landscape = 1 if aspect_ratio > 1.1 else 0
    is_square = 1 if 0.9 <= aspect_ratio <= 1.1 else 0

    # ========================================================================
    # Store results
    # ========================================================================
    results.append({
        'post_id': post_id,
        'account': account,
        'face_count': face_count,
        'has_faces': has_faces,
        'dominant_r': dominant_r,
        'dominant_g': dominant_g,
        'dominant_b': dominant_b,
        'brightness': brightness,
        'saturation': saturation,
        'color_variance': round(color_variance, 2),
        'aspect_ratio': round(aspect_ratio, 3),
        'resolution': resolution,
        'file_size_kb': round(file_size_kb, 2),
        'is_portrait': is_portrait,
        'is_landscape': is_landscape,
        'is_square': is_square
    })

# Create DataFrame
df_visual = pd.DataFrame(results)

# Save
output_path = 'data/processed/advanced_visual_features_multi_account.csv'
df_visual.to_csv(output_path, index=False)

print()
print("="*90)
print(" "*30 + "EXTRACTION COMPLETE!")
print("="*90)
print()
print(f"[SAVE] {output_path}")
print(f"[POSTS] {len(df_visual)} posts processed")
print()
print("[FEATURES] 15 advanced visual features:")
print("   - Face detection: face_count, has_faces")
print("   - Color analysis: dominant_r/g/b, brightness, saturation, color_variance")
print("   - Metadata: aspect_ratio, resolution, file_size_kb")
print("   - Orientation: is_portrait, is_landscape, is_square")
print()

# Summary statistics
print("[SUMMARY]")
print(f"   Posts with faces: {df_visual['has_faces'].sum()} ({df_visual['has_faces'].mean()*100:.1f}%)")
print(f"   Avg faces per post: {df_visual['face_count'].mean():.2f}")
print(f"   Avg brightness: {df_visual['brightness'].mean():.1f}")
print(f"   Avg saturation: {df_visual['saturation'].mean():.1f}")
print(f"   Portrait: {df_visual['is_portrait'].sum()} ({df_visual['is_portrait'].mean()*100:.1f}%)")
print(f"   Landscape: {df_visual['is_landscape'].sum()} ({df_visual['is_landscape'].mean()*100:.1f}%)")
print(f"   Square: {df_visual['is_square'].sum()} ({df_visual['is_square'].mean()*100:.1f}%)")
