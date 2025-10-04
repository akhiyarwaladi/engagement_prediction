#!/usr/bin/env python3
"""
Extract Data from Multiple Instagram Accounts
Support: fst_unja, univ.jambi, and other accounts
Handle both old and new JSON formats
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def extract_from_account(account_name):
    """Extract posts from a specific account"""
    gallery_path = Path(f'gallery-dl/instagram/{account_name}')

    if not gallery_path.exists():
        print(f"[WARN] Account {account_name} not found in gallery-dl/instagram/")
        return pd.DataFrame()

    # Find all JSON files
    json_files = list(gallery_path.glob('*.json'))

    print(f"\n[{account_name.upper()}] Found {len(json_files)} JSON files")

    posts = []

    for json_file in tqdm(json_files, desc=f"Processing {account_name}"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both formats: direct fields OR nested in 'node'
            node = data.get('node', data)

            # Extract post ID
            post_id = data.get('post_id') or data.get('media_id') or node.get('id', '')

            # Get caption - handle multiple formats
            caption = ''
            # Try 'description' first (new format from gallery-dl)
            if 'description' in data:
                caption = data['description'] or ''
            # Fallback to 'caption' (old format)
            elif 'caption' in data:
                caption = data['caption'] or ''
            # Fallback to nested format (very old)
            else:
                edge_caption = node.get('edge_media_to_caption', {})
                caption_edges = edge_caption.get('edges', [])
                caption = caption_edges[0]['node']['text'] if caption_edges else ''

            # Get media type
            typename = node.get('__typename', '')
            is_video = 1 if 'Video' in typename or data.get('video_url') else 0

            # Get engagement metrics - try multiple locations
            likes = (data.get('likes') or
                    node.get('edge_media_preview_like', {}).get('count', 0) or
                    node.get('edge_liked_by', {}).get('count', 0) or
                    0)

            comments = (data.get('comments') or
                       node.get('edge_media_to_comment', {}).get('count', 0) or
                       0)

            # Get timestamp
            timestamp = data.get('timestamp') or node.get('taken_at_timestamp', 0)
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
            else:
                # Try parsing date string
                date_str = data.get('date', '')
                if date_str:
                    try:
                        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except:
                        dt = datetime.now()
                else:
                    dt = datetime.now()

            # Count hashtags and mentions
            hashtag_count = caption.count('#')
            mention_count = caption.count('@')

            posts.append({
                'account': account_name,
                'post_id': post_id,
                'caption': caption,
                'caption_length': len(caption),
                'word_count': len(caption.split()),
                'hashtag_count': hashtag_count,
                'mention_count': mention_count,
                'is_video': is_video,
                'likes': int(likes),
                'comments': int(comments),
                'timestamp': timestamp if timestamp else 0,
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'year': dt.year,
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'day_of_week': dt.weekday(),
                'is_weekend': 1 if dt.weekday() >= 5 else 0,
            })

        except Exception as e:
            print(f"[ERROR] Failed to process {json_file.name}: {e}")
            continue

    df = pd.DataFrame(posts)
    print(f"[{account_name.upper()}] Extracted {len(df)} posts successfully")

    return df

def main():
    print("\n" + "="*80)
    print(" "*20 + "MULTI-ACCOUNT DATA EXTRACTION")
    print(" "*15 + "Extract Instagram Data from Multiple Accounts")
    print("="*80)

    # List of accounts to extract (8 UNJA accounts total)
    accounts = [
        'fst_unja',           # 1. Fakultas Sains & Teknologi
        'univ.jambi',         # 2. Universitas Jambi Official
        'fhunjaofficial',     # 3. Fakultas Hukum
        'bemfebunja',         # 4. BEM Fakultas Ekonomi & Bisnis
        'bemfkik.unja',       # 5. BEM Fak. Kedokteran & Ilmu Kesehatan
        'faperta.unja.official',  # 6. Fakultas Pertanian
        'himmajemen.unja',    # 7. Himpunan Mahasiswa Manajemen
        'fkipunja_official'   # 8. FKIP (Fak. Keguruan & Ilmu Pendidikan)
    ]

    all_posts = []

    for account in accounts:
        df = extract_from_account(account)
        if not df.empty:
            all_posts.append(df)

    if not all_posts:
        print("\n[ERROR] No data extracted from any account!")
        return

    # Combine all accounts
    combined_df = pd.concat(all_posts, ignore_index=True)

    print("\n" + "="*80)
    print("COMBINED DATASET SUMMARY")
    print("="*80)

    print(f"\n[TOTAL] {len(combined_df)} posts from {len(accounts)} accounts")

    # Per-account breakdown
    print("\n[BREAKDOWN] Posts per account:")
    for account in accounts:
        account_posts = combined_df[combined_df['account'] == account]
        videos = account_posts[account_posts['is_video'] == 1]
        print(f"   {account:15} {len(account_posts):4} posts ({len(videos):2} videos)")

    # Engagement statistics
    print(f"\n[ENGAGEMENT] Overall statistics:")
    print(f"   Total likes: {combined_df['likes'].sum():,}")
    print(f"   Mean likes: {combined_df['likes'].mean():.2f}")
    print(f"   Median likes: {combined_df['likes'].median():.2f}")
    print(f"   Std likes: {combined_df['likes'].std():.2f}")
    print(f"   Max likes: {combined_df['likes'].max():,}")

    # Video statistics
    videos = combined_df[combined_df['is_video'] == 1]
    photos = combined_df[combined_df['is_video'] == 0]
    print(f"\n[MEDIA] Media type breakdown:")
    print(f"   Photos: {len(photos)} ({len(photos)/len(combined_df)*100:.1f}%)")
    print(f"   Videos: {len(videos)} ({len(videos)/len(combined_df)*100:.1f}%)")
    if len(videos) > 0:
        print(f"   Video avg likes: {videos['likes'].mean():.2f}")
    if len(photos) > 0:
        print(f"   Photo avg likes: {photos['likes'].mean():.2f}")

    # Save combined dataset
    output_path = 'multi_account_dataset.csv'
    combined_df.to_csv(output_path, index=False)

    print(f"\n[SAVE] Combined dataset saved to: {output_path}")

    # Save per-account datasets
    for account in accounts:
        account_df = combined_df[combined_df['account'] == account]
        account_file = f'{account}_from_gallery_dl.csv'
        account_df.to_csv(account_file, index=False)
        print(f"       {account} dataset: {account_file}")

    print("\n" + "="*80)
    print("MULTI-ACCOUNT EXTRACTION COMPLETE!")
    print("="*80)
    print("")

if __name__ == '__main__':
    main()
