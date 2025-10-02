#!/usr/bin/env python3
"""
Extract metadata from gallery-dl JSON files (FIXED VERSION)
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

def extract_from_json(json_file):
    """Extract metadata from gallery-dl JSON"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get caption/description
    caption = data.get('description', '')

    # Extract hashtags and mentions
    hashtags = [word for word in caption.split() if word.startswith('#')]
    mentions = [word for word in caption.split() if word.startswith('@')]

    return {
        'post_id': data.get('media_id', ''),
        'shortcode': data.get('shortcode', ''),
        'url': data.get('post_url', f"https://www.instagram.com/p/{data.get('shortcode', '')}/"),
        'username': data.get('username', 'fst_unja'),
        'date': data.get('date', ''),
        'caption': caption,
        'likes': data.get('likes', 0),  # INI YANG BENAR!
        'comments': 0,  # Gallery-dl tidak punya field ini
        'is_video': data.get('video_url') is not None,
        'video_views': 0,  # Gallery-dl tidak track ini
        'location': '',
        'hashtags': ' '.join(hashtags),
        'hashtags_count': len(hashtags),
        'mentions': ' '.join(mentions),
        'mentions_count': len(mentions),
        'media_type': 'video' if data.get('video_url') else 'photo',
        'width': data.get('width', 0),
        'height': data.get('height', 0),
        'file_path': str(json_file).replace('.json', '')
    }

def main():
    print("=" * 70)
    print("EXTRACT FROM GALLERY-DL JSON (FIXED - WITH LIKES!)")
    print("=" * 70)

    # Find all JSON files
    json_folder = Path('gallery-dl/instagram/fst_unja')

    if not json_folder.exists():
        print(f"Error: {json_folder} not found!")
        return

    json_files = list(json_folder.glob('*.json'))
    print(f"\nFound {len(json_files)} JSON files")

    # Group by shortcode (to handle albums/carousels)
    posts_by_shortcode = defaultdict(list)

    for json_file in json_files:
        try:
            metadata = extract_from_json(json_file)
            shortcode = metadata['shortcode']

            # Remove album suffix (_123456) to get base shortcode
            base_shortcode = shortcode.split('_')[0] if '_' in shortcode else shortcode

            posts_by_shortcode[base_shortcode].append(metadata)

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    # Take first post from each shortcode group (deduplicate albums)
    unique_posts = []

    for shortcode, posts in posts_by_shortcode.items():
        # Take the post with most likes (usually the main post)
        best_post = max(posts, key=lambda x: x['likes'])
        unique_posts.append(best_post)

    # Sort by date (newest first)
    unique_posts.sort(key=lambda x: x['date'], reverse=True)

    # Save to CSV
    output_file = 'fst_unja_from_gallery_dl.csv'

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if unique_posts:
            writer = csv.DictWriter(f, fieldnames=unique_posts[0].keys())
            writer.writeheader()
            writer.writerows(unique_posts)

    # Statistics
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal JSON files: {len(json_files)}")
    print(f"Unique posts (after deduplication): {len(unique_posts)}")
    print(f"CSV saved to: {output_file}")

    # Engagement stats
    posts_with_likes = [p for p in unique_posts if p['likes'] > 0]
    total_likes = sum(p['likes'] for p in posts_with_likes)

    videos = sum(1 for p in unique_posts if p['is_video'])
    photos = len(unique_posts) - videos

    print(f"\nContent:")
    print(f"  Photos: {photos}")
    print(f"  Videos: {videos}")

    print(f"\nEngagement (from gallery-dl):")
    print(f"  Posts with likes: {len(posts_with_likes)}/{len(unique_posts)}")
    print(f"  Total likes: {total_likes:,}")

    if posts_with_likes:
        print(f"  Avg likes/post: {total_likes/len(posts_with_likes):.2f}")

    # Top 5 posts
    sorted_posts = sorted(posts_with_likes, key=lambda x: x['likes'], reverse=True)[:5]

    print(f"\nTop 5 Posts by Likes:")
    for i, p in enumerate(sorted_posts, 1):
        caption = p['caption'][:60].replace('\n', ' ') + '...'
        print(f"  {i}. {p['likes']:,} likes - {caption}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
