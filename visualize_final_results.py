#!/usr/bin/env python3
"""
Visualize Final Results: 8,610 Posts Analysis
==============================================
Create comprehensive visualizations comparing all models and phases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("\n" + "="*80)
print(" "*25 + "FINAL RESULTS VISUALIZATION")
print(" "*20 + "8,610 Posts - 3 Models Comparison")
print("="*80 + "\n")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path('docs/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. MODEL COMPARISON - MAE & R²
# ============================================================================

print("[PLOT 1] Model Performance Comparison...")

models = ['Baseline\n(9 feat)', 'Baseline+BERT\n(59 feat)', 'Full\n(67 feat)']
mae_values = [51.82, 77.76, 56.67]
r2_values = [0.8159, 0.7308, 0.8088]
features = [9, 59, 67]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MAE comparison
ax1 = axes[0]
bars1 = ax1.bar(models, mae_values, color=['#2ecc71', '#e74c3c', '#3498db'], alpha=0.7)
ax1.axhline(y=60, color='red', linestyle='--', linewidth=2, label='Target (MAE=60)')
ax1.axhline(y=94.54, color='orange', linestyle='--', linewidth=2, label='Previous Best (MAE=94.54)')
ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax1.set_title('Model MAE Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.set_ylim(0, 100)

# Add value labels on bars
for bar, val in zip(bars1, mae_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add champion marker
ax1.text(0, 48, '★ CHAMPION', ha='center', fontsize=12, fontweight='bold', color='#2ecc71')

# R² comparison
ax2 = axes[1]
bars2 = ax2.bar(models, r2_values, color=['#2ecc71', '#e74c3c', '#3498db'], alpha=0.7)
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Model R² Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1.0)

# Add value labels
for bar, val in zip(bars2, r2_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Feature count comparison
ax3 = axes[2]
bars3 = ax3.bar(models, features, color=['#2ecc71', '#e74c3c', '#3498db'], alpha=0.7)
ax3.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax3.set_title('Feature Complexity\n(Baseline Wins with Fewest)', fontsize=14, fontweight='bold')

# Add value labels
for bar, val in zip(bars3, features):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'final_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'final_model_comparison.png'}")

# ============================================================================
# 2. FEATURE IMPORTANCE (BASELINE CHAMPION)
# ============================================================================

print("\n[PLOT 2] Feature Importance (Champion Model)...")

features_names = ['month', 'hashtag_count', 'caption_length', 'word_count',
                  'hour', 'day_of_week', 'mention_count', 'is_weekend', 'is_video']
importances = [0.210431, 0.206256, 0.197384, 0.164812, 0.107199,
               0.067109, 0.035960, 0.008269, 0.002579]

fig, ax = plt.subplots(figsize=(12, 8))

# Sort by importance
sorted_idx = np.argsort(importances)
sorted_features = [features_names[i] for i in sorted_idx]
sorted_importances = [importances[i] for i in sorted_idx]

# Create horizontal bar chart
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
bars = ax.barh(sorted_features, sorted_importances, color=colors, alpha=0.8)

# Highlight top 3
for i, bar in enumerate(bars[-3:]):
    bar.set_edgecolor('red')
    bar.set_linewidth(2)

ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('Champion Model Feature Importance\n(Top 3: month, hashtag_count, caption_length)',
             fontsize=14, fontweight='bold')

# Add percentage labels
for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{val*100:.1f}%', ha='left', va='center', fontweight='bold')

# Add cumulative line for top features
ax.text(0.21, 8.3, 'Top 3 = 60.3%', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance_champion.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'feature_importance_champion.png'}")

# ============================================================================
# 3. PROGRESS TIMELINE
# ============================================================================

print("\n[PLOT 3] Progress Timeline...")

phases = ['Phase 0\n(348 posts)', 'Phase 1-2\n(1,949 posts)', 'Phase 3\n(8,610 posts)']
mae_timeline = [185.29, 94.54, 51.82]  # Estimated Phase 0
r2_timeline = [0.086, 0.206, 0.8159]   # Estimated
dataset_sizes = [348, 1949, 8610]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MAE timeline
ax1 = axes[0]
ax1.plot(phases, mae_timeline, marker='o', linewidth=3, markersize=12, color='#2ecc71')
ax1.fill_between(range(len(phases)), mae_timeline, alpha=0.3, color='#2ecc71')
ax1.axhline(y=60, color='red', linestyle='--', linewidth=2, label='Target (60)')
ax1.set_ylabel('MAE (likes)', fontsize=12, fontweight='bold')
ax1.set_title('MAE Improvement Over Time', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels
for i, (x, y) in enumerate(zip(phases, mae_timeline)):
    ax1.text(i, y + 5, f'{y:.1f}', ha='center', fontweight='bold', fontsize=11)

# R² timeline
ax2 = axes[1]
ax2.plot(phases, r2_timeline, marker='s', linewidth=3, markersize=12, color='#3498db')
ax2.fill_between(range(len(phases)), r2_timeline, alpha=0.3, color='#3498db')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('R² Improvement Over Time', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.grid(True, alpha=0.3)

# Add value labels
for i, (x, y) in enumerate(zip(phases, r2_timeline)):
    ax2.text(i, y + 0.03, f'{y:.4f}', ha='center', fontweight='bold', fontsize=11)

# Dataset size
ax3 = axes[2]
ax3.bar(phases, dataset_sizes, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.7)
ax3.set_ylabel('Dataset Size (posts)', fontsize=12, fontweight='bold')
ax3.set_title('Dataset Growth', fontsize=14, fontweight='bold')

# Add value labels
for i, (x, y) in enumerate(zip(phases, dataset_sizes)):
    ax3.text(i, y + 200, f'{y:,}', ha='center', fontweight='bold', fontsize=11)

# Add scale annotations
ax3.text(1, 5000, '5.6x', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax3.text(2, 6000, '4.4x', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'progress_timeline.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'progress_timeline.png'}")

# ============================================================================
# 4. ACCOUNT DISTRIBUTION
# ============================================================================

print("\n[PLOT 4] Account Distribution...")

accounts = ['univ.jambi', 'bemfkik.unja', 'faperta.unja.official', 'himmajemen.unja',
            'bemfebunja', 'fhunjaofficial', 'fst_unja', 'fkipunja_official']
post_counts = [1792, 1340, 1164, 1094, 1039, 813, 396, 182]
percentages = [c/sum(post_counts)*100 for c in post_counts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Bar chart
colors_accounts = plt.cm.Set3(np.linspace(0, 1, len(accounts)))
bars = ax1.barh(accounts, post_counts, color=colors_accounts, alpha=0.8)
ax1.set_xlabel('Number of Posts', fontsize=12, fontweight='bold')
ax1.set_title('Posts per Account', fontsize=14, fontweight='bold')

# Add value labels
for bar, val, pct in zip(bars, post_counts, percentages):
    width = bar.get_width()
    ax1.text(width + 30, bar.get_y() + bar.get_height()/2.,
             f'{val:,} ({pct:.1f}%)', ha='left', va='center', fontweight='bold')

# Pie chart
ax2.pie(post_counts, labels=accounts, autopct='%1.1f%%', startangle=90,
        colors=colors_accounts, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('Account Distribution\n(Total: 8,610 posts)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'account_distribution.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'account_distribution.png'}")

# ============================================================================
# 5. PERFORMANCE SUMMARY TABLE
# ============================================================================

print("\n[PLOT 5] Performance Summary Table...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Create summary table
summary_data = [
    ['Metric', 'Baseline', 'Baseline+BERT', 'Full Model'],
    ['Features', '9', '59', '67'],
    ['MAE (likes)', '51.82', '77.76', '56.67'],
    ['RMSE (likes)', '168.88', '204.19', '172.10'],
    ['R² Score', '0.8159', '0.7308', '0.8088'],
    ['Improvement vs Mean', '74.9%', '62.4%', '72.6%'],
    ['Status', '★ CHAMPION', 'Worst', 'Good']
]

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style champion column
for i in range(7):
    table[(i, 1)].set_facecolor('#d5f4e6')  # Light green

# Style worst column
for i in range(7):
    table[(i, 2)].set_facecolor('#fadbd8')  # Light red

# Style status row
table[(6, 1)].set_text_props(weight='bold', color='#2ecc71')
table[(6, 2)].set_text_props(weight='bold', color='#e74c3c')
table[(6, 3)].set_text_props(weight='bold', color='#3498db')

plt.title('Model Performance Summary\n8,610 Posts - October 4, 2025',
          fontsize=16, fontweight='bold', pad=20)

plt.savefig(output_dir / 'performance_summary_table.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'performance_summary_table.png'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)

print("\n[FILES CREATED]")
print("   1. final_model_comparison.png - 3 models MAE/R²/Features comparison")
print("   2. feature_importance_champion.png - Champion model feature importance")
print("   3. progress_timeline.png - MAE/R²/Dataset evolution")
print("   4. account_distribution.png - 8 accounts breakdown")
print("   5. performance_summary_table.png - Complete metrics table")

print(f"\n[LOCATION] All files saved to: {output_dir.absolute()}")

print("\n[KEY FINDINGS]")
print("   • Baseline (9 features) is CHAMPION: MAE=51.82, R²=0.8159")
print("   • 45% improvement vs previous best (1,949 posts)")
print("   • Top 3 features: month (21%), hashtags (20.6%), caption_length (19.7%)")
print("   • Simplicity wins at 8,610 posts scale!")

print("\n" + "="*80 + "\n")
