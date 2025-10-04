#!/usr/bin/env python3
"""
Compare All Phases: Single Account vs Multi-Account
Comprehensive comparison of model performance across all experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("\n" + "="*80)
print(" "*25 + "MODEL COMPARISON")
print(" "*15 + "Single Account vs Multi-Account Analysis")
print("="*80 + "\n")

# Single account results (348 posts)
single_account = pd.DataFrame([
    {'phase': 'Phase 0\nBaseline', 'model': 'Baseline RF', 'posts': 348, 'features': 9, 'mae': 185.29, 'r2': 0.086},
    {'phase': 'Phase 1\nLog+Interact', 'model': 'RF + Log Transform', 'posts': 348, 'features': 14, 'mae': 115.17, 'r2': 0.090},
    {'phase': 'Phase 2\nNLP+Ensemble', 'model': 'Ensemble + NLP', 'posts': 348, 'features': 28, 'mae': 109.42, 'r2': 0.200},
    {'phase': 'Phase 4a\nIndoBERT', 'model': 'Baseline + BERT', 'posts': 348, 'features': 59, 'mae': 98.94, 'r2': 0.206},
    {'phase': 'Phase 4b\nMultimodal', 'model': 'BERT + ViT', 'posts': 348, 'features': 109, 'mae': 111.28, 'r2': 0.234},
])

# Multi-account results (1,949 posts)
multi_account = pd.DataFrame([
    {'phase': 'Multi-Account\nBaseline', 'model': 'Baseline (9 features)', 'posts': 1949, 'features': 9, 'mae': 125.58, 'r2': 0.6746},
    {'phase': 'Multi-Account\nBERT', 'model': 'Baseline + BERT', 'posts': 1949, 'features': 59, 'mae': 94.54, 'r2': 0.7234},
    {'phase': 'Multi-Account\nBERT+NIMA', 'model': 'Baseline + BERT + NIMA', 'posts': 1949, 'features': 67, 'mae': 108.01, 'r2': 0.7013},
])

# Combine
all_results = pd.concat([single_account, multi_account], ignore_index=True)

# Print summary table
print("[SUMMARY TABLE] All Model Performance\n")
print(f"{'Phase':<25} {'Posts':<8} {'Features':<10} {'MAE':>10} {'R²':>10}")
print("-" * 80)

for _, row in all_results.iterrows():
    print(f"{row['phase']:<25} {row['posts']:<8} {row['features']:<10} {row['mae']:>10.2f} {row['r2']:>10.4f}")

# Find best models
best_mae = all_results.loc[all_results['mae'].idxmin()]
best_r2 = all_results.loc[all_results['r2'].idxmax()]

print("\n" + "="*80)
print("[CHAMPIONS]")
print(f"   Best MAE: {best_mae['phase']} (MAE={best_mae['mae']:.2f}, {best_mae['posts']} posts)")
print(f"   Best R²:  {best_r2['phase']} (R²={best_r2['r2']:.4f}, {best_r2['posts']} posts)")

# Calculate improvements
old_champion_mae = 135.21  # Phase 2 RFE model (not in table but documented)
new_champion_mae = best_mae['mae']
mae_improvement = (old_champion_mae - new_champion_mae) / old_champion_mae * 100

old_champion_r2 = 0.4705
new_champion_r2 = best_r2['r2']
r2_improvement = (new_champion_r2 - old_champion_r2) / old_champion_r2 * 100

print(f"\n[IMPROVEMENTS vs Previous Champion (RFE 75 features)]")
print(f"   MAE: {old_champion_mae:.2f} -> {new_champion_mae:.2f} ({mae_improvement:+.1f}%)")
print(f"   R²:  {old_champion_r2:.4f} -> {new_champion_r2:.4f} ({r2_improvement:+.1f}%)")

# Key insights
print("\n" + "="*80)
print("[KEY INSIGHTS]")
print("\n1. DATA SCALING IMPACT:")
print(f"   Baseline (348 posts): R² = 0.086")
print(f"   Baseline (1,949 posts): R² = 0.6746")
print(f"   Improvement: +684% just from more data!")

print("\n2. BERT DOMINANCE:")
print(f"   Single-account BERT (348 posts): MAE = 98.94")
print(f"   Multi-account BERT (1,949 posts): MAE = 94.54")
print(f"   Multi-account is 4.5% better with 5.6x more data")

print("\n3. NIMA AESTHETICS:")
print(f"   BERT alone (1,949 posts): MAE = 94.54")
print(f"   BERT + NIMA (1,949 posts): MAE = 108.01")
print(f"   NIMA degraded performance by 14.2%")
print(f"   -> Visual aesthetics not critical for academic Instagram")

print("\n4. FEATURE EFFICIENCY:")
single_best = single_account.loc[single_account['mae'].idxmin()]
multi_best = multi_account.loc[multi_account['mae'].idxmin()]
print(f"   Single-account best: {single_best['features']} features -> MAE {single_best['mae']:.2f}")
print(f"   Multi-account best: {multi_best['features']} features -> MAE {multi_best['mae']:.2f}")
print(f"   Same feature count, better data = better model!")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: MAE comparison
ax1 = axes[0]
colors = ['#e74c3c' if row['posts'] == 348 else '#2ecc71' for _, row in all_results.iterrows()]
bars1 = ax1.bar(range(len(all_results)), all_results['mae'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Model Phase', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax1.set_title('MAE Comparison: Single vs Multi-Account', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(all_results)))
ax1.set_xticklabels(all_results['phase'], rotation=45, ha='right', fontsize=9)
ax1.axhline(y=94.54, color='green', linestyle='--', linewidth=2, label='Best MAE (94.54)')
ax1.axhline(y=135.21, color='red', linestyle='--', linewidth=2, label='Old Champion (135.21)')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
             f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# Plot 2: R² comparison
ax2 = axes[1]
bars2 = ax2.bar(range(len(all_results)), all_results['r2'], color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Model Phase', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('R² Comparison: Single vs Multi-Account', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(all_results)))
ax2.set_xticklabels(all_results['phase'], rotation=45, ha='right', fontsize=9)
ax2.axhline(y=0.7234, color='green', linestyle='--', linewidth=2, label='Best R² (0.7234)')
ax2.axhline(y=0.4705, color='red', linestyle='--', linewidth=2, label='Old Champion (0.4705)')
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Legend
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                       markersize=10, label='Single Account (348 posts)', alpha=0.7)
green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
                         markersize=10, label='Multi-Account (1,949 posts)', alpha=0.7)
fig.legend(handles=[red_patch, green_patch], loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 0.98), fontsize=11, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('docs/figures/multi_account_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n[PLOT] Saved to: docs/figures/multi_account_comparison.png")

# Feature vs Performance scatter
fig2, ax = plt.subplots(figsize=(12, 8))
for dataset, marker, label in [('348', 'o', 'Single Account (348 posts)'),
                                ('1949', 's', 'Multi-Account (1,949 posts)')]:
    df_subset = all_results[all_results['posts'] == int(dataset)]
    ax.scatter(df_subset['features'], df_subset['mae'],
               s=200, marker=marker, alpha=0.7, label=label, edgecolors='black')

    # Add labels
    for _, row in df_subset.iterrows():
        ax.annotate(row['phase'],
                   (row['features'], row['mae']),
                   textcoords="offset points", xytext=(0,10),
                   ha='center', fontsize=8, weight='bold')

ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax.set_title('Features vs Performance: Diminishing Returns Analysis', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, frameon=True)
ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Lower MAE is better

plt.tight_layout()
plt.savefig('docs/figures/features_vs_performance.png', dpi=300, bbox_inches='tight')
print(f"[PLOT] Saved to: docs/figures/features_vs_performance.png")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80 + "\n")

# Save summary CSV
all_results.to_csv('experiments/all_phases_comparison.csv', index=False)
print(f"[SAVE] Comprehensive results: experiments/all_phases_comparison.csv\n")
