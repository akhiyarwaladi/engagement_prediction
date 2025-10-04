#!/usr/bin/env python3
"""
Extract Advanced Video Features
Temporal analysis, scene detection, motion patterns, audio features
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_temporal_features(video_path):
    """Extract advanced temporal features from video"""
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0

    if frame_count == 0 or fps == 0:
        return get_zero_video_features()

    # Sample frames for analysis
    sample_interval = max(1, int(frame_count / 30))  # Max 30 frames
    frames = []
    frame_indices = []

    for i in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_indices.append(i)

    cap.release()

    if len(frames) < 2:
        return get_zero_video_features()

    # Convert to grayscale for analysis
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # 1. Motion analysis (frame differences)
    motion_scores = []
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
        motion_scores.append(np.mean(diff))

    motion_mean = np.mean(motion_scores) if motion_scores else 0
    motion_std = np.std(motion_scores) if motion_scores else 0
    motion_max = np.max(motion_scores) if motion_scores else 0

    # 2. Scene changes (large motion spikes)
    motion_threshold = motion_mean + 2 * motion_std if motion_std > 0 else motion_mean
    scene_changes = sum(1 for m in motion_scores if m > motion_threshold)
    scene_change_rate = scene_changes / duration if duration > 0 else 0

    # 3. Brightness variation over time
    brightness_values = [np.mean(f) for f in gray_frames]
    brightness_mean = np.mean(brightness_values)
    brightness_std = np.std(brightness_values)
    brightness_range = np.max(brightness_values) - np.min(brightness_values)

    # 4. Color variation over time
    color_std_values = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_std_values.append(np.std(hsv[:, :, 1]))  # Saturation std

    color_variation = np.mean(color_std_values)

    # 5. Edge density variation (visual complexity changes)
    edge_densities = []
    for gray in gray_frames:
        edges = cv2.Canny(gray, 50, 150)
        edge_densities.append(np.sum(edges) / edges.size)

    edge_complexity_mean = np.mean(edge_densities)
    edge_complexity_std = np.std(edge_densities)

    # 6. Optical flow magnitude (advanced motion)
    if len(gray_frames) >= 2:
        flow_magnitudes = []
        for i in range(1, min(len(gray_frames), 10)):  # Check first 10 transitions
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i-1], gray_frames[i],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(np.mean(magnitude))

        optical_flow_mean = np.mean(flow_magnitudes) if flow_magnitudes else 0
        optical_flow_max = np.max(flow_magnitudes) if flow_magnitudes else 0
    else:
        optical_flow_mean = 0
        optical_flow_max = 0

    # 7. Aspect ratio and resolution features
    aspect_ratio = width / height if height > 0 else 1.0
    resolution = width * height
    resolution_category = 'low' if resolution < 500000 else ('medium' if resolution < 2000000 else 'high')

    # 8. Pacing analysis (motion acceleration)
    if len(motion_scores) >= 3:
        motion_acceleration = []
        for i in range(1, len(motion_scores)):
            acc = motion_scores[i] - motion_scores[i-1]
            motion_acceleration.append(abs(acc))
        motion_pacing = np.mean(motion_acceleration)
    else:
        motion_pacing = 0

    return {
        # Basic
        'video_duration': float(duration),
        'video_fps': float(fps),
        'video_frame_count': float(frame_count),
        'video_width': float(width),
        'video_height': float(height),
        'video_aspect_ratio': float(aspect_ratio),
        'video_resolution': float(resolution),

        # Motion features
        'video_motion_mean': float(motion_mean),
        'video_motion_std': float(motion_std),
        'video_motion_max': float(motion_max),
        'video_motion_pacing': float(motion_pacing),

        # Scene features
        'video_scene_changes': float(scene_changes),
        'video_scene_change_rate': float(scene_change_rate),

        # Brightness features
        'video_brightness_mean': float(brightness_mean),
        'video_brightness_std': float(brightness_std),
        'video_brightness_range': float(brightness_range),

        # Color features
        'video_color_variation': float(color_variation),

        # Complexity features
        'video_edge_complexity_mean': float(edge_complexity_mean),
        'video_edge_complexity_std': float(edge_complexity_std),

        # Optical flow
        'video_optical_flow_mean': float(optical_flow_mean),
        'video_optical_flow_max': float(optical_flow_max),
    }

def get_zero_video_features():
    """Return zero features for non-video posts"""
    return {
        'video_duration': 0.0,
        'video_fps': 0.0,
        'video_frame_count': 0.0,
        'video_width': 0.0,
        'video_height': 0.0,
        'video_aspect_ratio': 0.0,
        'video_resolution': 0.0,
        'video_motion_mean': 0.0,
        'video_motion_std': 0.0,
        'video_motion_max': 0.0,
        'video_motion_pacing': 0.0,
        'video_scene_changes': 0.0,
        'video_scene_change_rate': 0.0,
        'video_brightness_mean': 0.0,
        'video_brightness_std': 0.0,
        'video_brightness_range': 0.0,
        'video_color_variation': 0.0,
        'video_edge_complexity_mean': 0.0,
        'video_edge_complexity_std': 0.0,
        'video_optical_flow_mean': 0.0,
        'video_optical_flow_max': 0.0,
    }

def main():
    print("\n" + "="*80)
    print(" "*20 + "ADVANCED VIDEO FEATURES EXTRACTION")
    print(" "*15 + "Temporal, Motion, Scene, Optical Flow Analysis")
    print("="*80)

    # Load baseline dataset
    baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
    print(f"\n[DATA] Loaded {len(baseline_df)} posts")

    videos = baseline_df[baseline_df['is_video'] == 1]
    print(f"   Videos: {len(videos)} ({len(videos)/len(baseline_df)*100:.1f}%)")

    # Find video files
    gallery_path = Path('gallery-dl/instagram/fst_unja')

    video_features_list = []

    print("\n[EXTRACT] Processing videos...")
    for idx, row in tqdm(baseline_df.iterrows(), total=len(baseline_df)):
        post_id = row['post_id']
        is_video = row['is_video']

        if not is_video:
            # Non-video: zero features
            features = {'post_id': post_id, **get_zero_video_features()}
        else:
            # Find video file
            video_files = list(gallery_path.glob(f"{post_id}.*"))
            video_files = [f for f in video_files if f.suffix.lower() in ['.mp4', '.mov', '.avi']]

            if not video_files:
                print(f"[WARN] Video file not found for {post_id}")
                features = {'post_id': post_id, **get_zero_video_features()}
            else:
                video_path = video_files[0]
                video_features = extract_temporal_features(video_path)
                features = {'post_id': post_id, **video_features}

        video_features_list.append(features)

    # Create DataFrame
    video_df = pd.DataFrame(video_features_list)

    # Save
    output_path = 'data/processed/advanced_video_features.csv'
    video_df.to_csv(output_path, index=False)

    print(f"\n[SAVE] Advanced video features saved: {output_path}")
    print(f"       Total posts: {len(video_df)}")
    print(f"       Total features: {len(video_df.columns) - 1}")

    # Summary statistics for videos only
    video_data = video_df[video_df['video_duration'] > 0]
    print(f"\n[SUMMARY] Video Feature Statistics ({len(video_data)} videos):")
    print("-" * 80)

    for col in video_df.columns:
        if col == 'post_id':
            continue
        values = video_data[col]
        if len(values) > 0:
            print(f"  {col:35} mean={values.mean():8.2f} std={values.std():8.2f} "
                  f"min={values.min():8.2f} max={values.max():8.2f}")

    print("\n" + "="*80)
    print("ADVANCED VIDEO FEATURE EXTRACTION COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
