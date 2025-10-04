#!/usr/bin/env python3
"""
Aspect Ratio Distribution Analysis
===================================

BREAKTHROUGH: Aspect ratio is the BEST single quality feature (+3.43% MAE improvement)!

Question: What aspect ratios get the most engagement?
- Square (1:1 = 1.0)?
- Landscape (16:9 = 1.78)?
- Portrait (4:5 = 0.8)?
- Instagram optimal (4:5 = 0.8)?

Let's find the OPTIMAL aspect ratio for @fst_unja!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("\n" + "=" * 90)
print(" " * 20 + "ASPECT RATIO DISTRIBUTION ANALYSIS")
print(" " * 15 + "Finding the Optimal Aspect Ratio for Engagement")
print("=" * 90)

# Load data
print("\n[DATA] Loading datasets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
enhanced_df = pd.read_csv('data/processed/enhanced_visual_features.csv')

# Merge
df = baseline_df.merge(enhanced_df[['post_id', 'aspect_ratio', 'sharpness', 'contrast']],
                       left_on='post_id', right_on='post_id', how='left')

# Filter photos only (videos have aspect_ratio=0)
photos_df = df[df['is_video'] == 0].copy()

print(f"   Total posts: {len(df)}")
print(f"   Photos: {len(photos_df)}")
print(f"   Videos: {len(df) - len(photos_df)}")

# Aspect ratio statistics
print("\n[STATS] Aspect Ratio Statistics (Photos Only):")
print(f"   Mean: {photos_df['aspect_ratio'].mean():.3f}")
print(f"   Median: {photos_df['aspect_ratio'].median():.3f}")
print(f"   Std: {photos_df['aspect_ratio'].std():.3f}")
print(f"   Min: {photos_df['aspect_ratio'].min():.3f}")
print(f"   Max: {photos_df['aspect_ratio'].max():.3f}")

# Define aspect ratio categories
def categorize_aspect_ratio(ar):
    """Categorize aspect ratio into Instagram formats."""
    if ar == 0:
        return 'Video'
    elif ar < 0.85:
        return 'Portrait (4:5, 0.8)'  # Instagram portrait
    elif 0.85 <= ar < 1.15:
        return 'Square (1:1, 1.0)'  # Instagram square
    elif 1.15 <= ar < 1.5:
        return 'Slightly Wide (1.2-1.4)'
    else:
        return 'Landscape (16:9, 1.78)'  # Instagram landscape

photos_df['ar_category'] = photos_df['aspect_ratio'].apply(categorize_aspect_ratio)

# Engagement by aspect ratio category
print("\n[ANALYSIS] Engagement by Aspect Ratio Category:")
print("-" * 90)
print(f"{'Category':<25} | {'Count':>8} | {'Avg Likes':>10} | {'Med Likes':>10} | {'Std':>10}")
print("-" * 90)

category_stats = []
for cat in ['Portrait (4:5, 0.8)', 'Square (1:1, 1.0)', 'Slightly Wide (1.2-1.4)', 'Landscape (16:9, 1.78)']:
    cat_df = photos_df[photos_df['ar_category'] == cat]
    if len(cat_df) > 0:
        avg_likes = cat_df['likes'].mean()
        med_likes = cat_df['likes'].median()
        std_likes = cat_df['likes'].std()
        count = len(cat_df)

        category_stats.append({
            'category': cat,
            'count': count,
            'avg_likes': avg_likes,
            'med_likes': med_likes,
            'std_likes': std_likes
        })

        print(f"{cat:<25} | {count:>8} | {avg_likes:>10.1f} | {med_likes:>10.1f} | {std_likes:>10.1f}")

print("-" * 90)

# Find best category
best_cat = max(category_stats, key=lambda x: x['avg_likes'])
worst_cat = min(category_stats, key=lambda x: x['avg_likes'])

print(f"\n[BEST] Highest engagement: {best_cat['category']}")
print(f"   Avg likes: {best_cat['avg_likes']:.1f}")
print(f"   Count: {best_cat['count']} photos")

print(f"\n[WORST] Lowest engagement: {worst_cat['category']}")
print(f"   Avg likes: {worst_cat['avg_likes']:.1f}")
print(f"   Count: {worst_cat['count']} photos")

improvement = ((best_cat['avg_likes'] - worst_cat['avg_likes']) / worst_cat['avg_likes']) * 100
print(f"\n[IMPACT] Using {best_cat['category']} vs {worst_cat['category']}: +{improvement:.1f}% more likes!")

# Correlation analysis
print("\n[CORRELATION] Aspect Ratio vs Likes:")
correlation = photos_df[['aspect_ratio', 'likes']].corr().iloc[0, 1]
print(f"   Pearson correlation: {correlation:.4f}")

if abs(correlation) < 0.1:
    print("   Weak correlation (linear relationship minimal)")
elif abs(correlation) < 0.3:
    print("   Moderate correlation")
else:
    print("   Strong correlation!")

# Optimal range analysis
print("\n[OPTIMAL] Finding optimal aspect ratio range:")

# Create bins
bins = [0, 0.85, 1.15, 1.5, 3.0]
labels = ['<0.85 (Portrait)', '0.85-1.15 (Square)', '1.15-1.5 (Wide)', '>1.5 (Landscape)']
photos_df['ar_bin'] = pd.cut(photos_df['aspect_ratio'], bins=bins, labels=labels)

print("-" * 90)
print(f"{'Range':<25} | {'Count':>8} | {'Avg Likes':>10} | {'% of Total':>10}")
print("-" * 90)

total_photos = len(photos_df)
for label in labels:
    bin_df = photos_df[photos_df['ar_bin'] == label]
    if len(bin_df) > 0:
        avg_likes = bin_df['likes'].mean()
        pct = (len(bin_df) / total_photos) * 100
        print(f"{label:<25} | {len(bin_df):>8} | {avg_likes:>10.1f} | {pct:>9.1f}%")

print("-" * 90)

# Top performers analysis
print("\n[TOP PERFORMERS] Aspect ratio of top 20% most-liked posts:")
top_20_pct = photos_df.nlargest(int(len(photos_df) * 0.2), 'likes')
top_ar_mean = top_20_pct['aspect_ratio'].mean()
top_ar_median = top_20_pct['aspect_ratio'].median()

print(f"   Mean aspect ratio: {top_ar_mean:.3f}")
print(f"   Median aspect ratio: {top_ar_median:.3f}")
print(f"   Most common category: {top_20_pct['ar_category'].mode().values[0]}")

# Bottom performers
bottom_20_pct = photos_df.nsmallest(int(len(photos_df) * 0.2), 'likes')
bottom_ar_mean = bottom_20_pct['aspect_ratio'].mean()
bottom_ar_median = bottom_20_pct['aspect_ratio'].median()

print(f"\n[BOTTOM PERFORMERS] Aspect ratio of bottom 20% least-liked posts:")
print(f"   Mean aspect ratio: {bottom_ar_mean:.3f}")
print(f"   Median aspect ratio: {bottom_ar_median:.3f}")
print(f"   Most common category: {bottom_20_pct['ar_category'].mode().values[0]}")

# Recommendations
print("\n" + "=" * 90)
print("RECOMMENDATIONS FOR @FST_UNJA")
print("=" * 90)

print(f"\n1. OPTIMAL ASPECT RATIO:")
print(f"   Target: {best_cat['category']}")
print(f"   Expected engagement: {best_cat['avg_likes']:.1f} likes avg")
print(f"   Current distribution: {(best_cat['count']/total_photos)*100:.1f}% of posts")

print(f"\n2. AVOID:")
print(f"   Avoid: {worst_cat['category']}")
print(f"   Lower engagement: {worst_cat['avg_likes']:.1f} likes avg")
print(f"   Impact: -{improvement:.1f}% compared to optimal")

print(f"\n3. FEED VISIBILITY OPTIMIZATION:")
if best_cat['category'] == 'Square (1:1, 1.0)':
    print("   Square (1:1) format:")
    print("   [OK] Maximum feed visibility")
    print("   [OK] No cropping in feed")
    print("   [OK] Instagram's recommended format")
elif best_cat['category'] == 'Portrait (4:5, 0.8)':
    print("   Portrait (4:5) format:")
    print("   [OK] Takes more vertical space in feed")
    print("   [OK] Instagram's portrait-optimized format")
    print("   [OK] Better for mobile viewing")
else:
    print(f"   {best_cat['category']}:")
    print("   Note: Instagram crops non-standard aspect ratios")

print(f"\n4. CONSISTENCY:")
current_distribution = photos_df['ar_category'].value_counts()
most_common = current_distribution.index[0]
most_common_pct = (current_distribution.values[0] / len(photos_df)) * 100

print(f"   Current: {most_common_pct:.1f}% of posts use {most_common}")
if most_common == best_cat['category']:
    print("   [OK] Already using optimal format! Keep it up!")
else:
    print(f"   [\!] Consider shifting to {best_cat['category']} for better engagement")

# Visualizations
print("\n[VIZ] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Aspect Ratio Analysis - Instagram Engagement', fontsize=16, fontweight='bold')

# Plot 1: Distribution
ax1 = axes[0, 0]
photos_df['aspect_ratio'].hist(bins=30, ax=ax1, edgecolor='black', alpha=0.7)
ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Square (1:1)')
ax1.axvline(x=0.8, color='green', linestyle='--', linewidth=2, label='Portrait (4:5)')
ax1.axvline(x=1.78, color='blue', linestyle='--', linewidth=2, label='Landscape (16:9)')
ax1.set_xlabel('Aspect Ratio', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Aspect Ratio Distribution', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Engagement by category
ax2 = axes[0, 1]
cat_names = [c['category'] for c in category_stats]
cat_likes = [c['avg_likes'] for c in category_stats]
colors = ['green' if c == best_cat['category'] else 'red' if c == worst_cat['category'] else 'gray' for c in cat_names]

bars = ax2.bar(range(len(cat_names)), cat_likes, color=colors, alpha=0.7)
ax2.set_xticks(range(len(cat_names)))
ax2.set_xticklabels([c.replace(' ', '\n') for c in cat_names], fontsize=8, rotation=0)
ax2.set_ylabel('Average Likes', fontweight='bold')
ax2.set_title('Engagement by Aspect Ratio Category', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, cat_likes)):
    ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}',
            ha='center', va='bottom', fontweight='bold')

# Plot 3: Scatter plot
ax3 = axes[1, 0]
ax3.scatter(photos_df['aspect_ratio'], photos_df['likes'], alpha=0.5, s=30)
ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Square')
ax3.axvline(x=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Portrait')
ax3.set_xlabel('Aspect Ratio', fontweight='bold')
ax3.set_ylabel('Likes', fontweight='bold')
ax3.set_title('Aspect Ratio vs Engagement (Scatter)', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Box plot by category
ax4 = axes[1, 1]
category_order = [c['category'] for c in sorted(category_stats, key=lambda x: x['avg_likes'], reverse=True)]
photos_df_filtered = photos_df[photos_df['ar_category'].isin(category_order)]

sns.boxplot(data=photos_df_filtered, y='ar_category', x='likes', order=category_order, ax=ax4, palette='Set2')
ax4.set_ylabel('Aspect Ratio Category', fontweight='bold')
ax4.set_xlabel('Likes', fontweight='bold')
ax4.set_title('Engagement Distribution by Category', fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()

output_path = 'docs/figures/aspect_ratio_analysis.png'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SAVE] Chart saved to: {output_path}")

# Save detailed results
results_df = pd.DataFrame(category_stats)
results_df.to_csv('experiments/aspect_ratio_analysis_results.csv', index=False)
print(f"[SAVE] Results saved to: experiments/aspect_ratio_analysis_results.csv")

print("\n" + "=" * 90)
print("ASPECT RATIO ANALYSIS COMPLETE!")
print("=" * 90)

print(f"\n[KEY FINDING] {best_cat['category']} gets {improvement:.1f}% more likes than {worst_cat['category']}!")
print(f"[RECOMMENDATION] @fst_unja should use {best_cat['category']} format for maximum engagement!")
print("")
