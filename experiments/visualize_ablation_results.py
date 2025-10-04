#!/usr/bin/env python3
"""
Visualize Ablation Study Results
=================================

Create charts to show which features help and which don't
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
results_df = pd.read_csv('experiments/feature_ablation_results.csv')

# Calculate improvements vs baseline
baseline_mae = results_df[results_df['name'] == 'Text-Only Baseline']['mae'].values[0]
baseline_r2 = results_df[results_df['name'] == 'Text-Only Baseline']['r2'].values[0]

results_df['mae_improvement'] = ((baseline_mae - results_df['mae']) / baseline_mae) * 100
results_df['r2_improvement'] = ((results_df['r2'] - baseline_r2) / baseline_r2) * 100

# Remove baseline from comparison
comparison_df = results_df[results_df['name'] != 'Text-Only Baseline'].copy()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Ablation Study Results - What Works and What Doesn\'t',
             fontsize=16, fontweight='bold')

# Plot 1: MAE Improvement
ax1 = axes[0, 0]
colors = ['green' if x > 0 else 'red' for x in comparison_df['mae_improvement']]
bars1 = ax1.barh(comparison_df['name'], comparison_df['mae_improvement'], color=colors, alpha=0.7)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.set_xlabel('MAE Improvement vs Text-Only (%)', fontweight='bold')
ax1.set_title('MAE: Positive = Better Accuracy', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (name, val) in enumerate(zip(comparison_df['name'], comparison_df['mae_improvement'])):
    color = 'green' if val > 0 else 'red'
    ax1.text(val, i, f'{val:+.1f}%', va='center', ha='left' if val > 0 else 'right',
             fontweight='bold', color=color)

# Plot 2: R² Improvement
ax2 = axes[0, 1]
colors = ['green' if x > 0 else 'red' for x in comparison_df['r2_improvement']]
bars2 = ax2.barh(comparison_df['name'], comparison_df['r2_improvement'], color=colors, alpha=0.7)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('R² Improvement vs Text-Only (%)', fontweight='bold')
ax2.set_title('R²: Positive = Better Pattern Understanding', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (name, val) in enumerate(zip(comparison_df['name'], comparison_df['r2_improvement'])):
    color = 'green' if val > 0 else 'red'
    ax2.text(val, i, f'{val:+.1f}%', va='center', ha='left' if val > 0 else 'right',
             fontweight='bold', color=color)

# Plot 3: MAE vs R² Scatter
ax3 = axes[1, 0]

# Color by group
group_colors = {
    'baseline': 'blue',
    'face': 'red',
    'text': 'orange',
    'color': 'purple',
    'quality': 'green',
    'video': 'brown',
    'combined': 'cyan'
}

for group in results_df['group'].unique():
    group_data = results_df[results_df['group'] == group]
    ax3.scatter(group_data['mae'], group_data['r2'],
               s=100, alpha=0.6, label=group, color=group_colors.get(group, 'gray'))

ax3.set_xlabel('MAE (lower is better)', fontweight='bold')
ax3.set_ylabel('R² (higher is better)', fontweight='bold')
ax3.set_title('MAE vs R² Trade-off', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Add annotations for best models
best_mae = results_df.loc[results_df['mae'].idxmin()]
best_r2 = results_df.loc[results_df['r2'].idxmax()]

ax3.annotate('Best MAE', xy=(best_mae['mae'], best_mae['r2']),
            xytext=(best_mae['mae']+2, best_mae['r2']-0.01),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontweight='bold', color='green')

ax3.annotate('Best R²', xy=(best_r2['mae'], best_r2['r2']),
            xytext=(best_r2['mae']+2, best_r2['r2']+0.01),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontweight='bold', color='blue')

# Plot 4: Feature Groups Ranking
ax4 = axes[1, 1]

# Get single-group results only
single_groups = comparison_df[comparison_df['group'].isin(['face', 'text', 'color', 'quality', 'video'])].copy()
single_groups = single_groups.sort_values('mae')

y_pos = range(len(single_groups))
colors_rank = ['green' if x < baseline_mae else 'red' for x in single_groups['mae']]

ax4.barh(y_pos, single_groups['mae'], color=colors_rank, alpha=0.7)
ax4.set_yticks(y_pos)
ax4.set_yticklabels([name.replace(' Only', '') for name in single_groups['name']])
ax4.axvline(x=baseline_mae, color='blue', linestyle='--', linewidth=2, label='Text-Only Baseline')
ax4.set_xlabel('MAE (likes)', fontweight='bold')
ax4.set_title('Feature Group Ranking (Lower MAE = Better)', fontweight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

# Add value labels
for i, val in enumerate(single_groups['mae']):
    ax4.text(val, i, f'{val:.1f}', va='center', ha='left', fontweight='bold')

plt.tight_layout()

# Save figure
output_path = 'docs/figures/feature_ablation_results.png'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[SAVE] Chart saved to: {output_path}")

# Create summary table figure
fig2, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Rank', 'Feature Group', 'MAE', 'R²', 'MAE Improve', 'R² Improve', 'Verdict'])

for i, row in comparison_df.iterrows():
    mae_symbol = '+' if row['mae_improvement'] > 0 else ''
    r2_symbol = '+' if row['r2_improvement'] > 0 else ''

    verdict = ''
    if row['mae_improvement'] > 0 and row['r2_improvement'] > 0:
        verdict = 'GOOD'
    elif row['mae_improvement'] < -2:
        verdict = 'BAD'
    else:
        verdict = 'NEUTRAL'

    table_data.append([
        '',
        row['name'],
        f"{row['mae']:.2f}",
        f"{row['r2']:.4f}",
        f"{mae_symbol}{row['mae_improvement']:.1f}%",
        f"{r2_symbol}{row['r2_improvement']:.1f}%",
        verdict
    ])

# Add baseline reference
table_data.append(['', '---', '---', '---', '---', '---', '---'])
table_data.append(['REF', 'Text-Only Baseline', f"{baseline_mae:.2f}", f"{baseline_r2:.4f}", '0.0%', '0.0%', 'BASELINE'])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.08, 0.3, 0.12, 0.12, 0.13, 0.12, 0.13])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(7):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color verdict column
for i in range(1, len(table_data)):
    if i < len(table_data) - 2:  # Skip separator and baseline
        verdict = table_data[i][6]
        if verdict == 'GOOD':
            table[(i, 6)].set_facecolor('#90EE90')
        elif verdict == 'BAD':
            table[(i, 6)].set_facecolor('#FFB6C1')
        else:
            table[(i, 6)].set_facecolor('#FFFFE0')

plt.title('Feature Ablation Study - Complete Results Summary',
         fontsize=14, fontweight='bold', pad=20)

# Save summary table
output_path2 = 'docs/figures/feature_ablation_table.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"[SAVE] Table saved to: {output_path2}")

print("\n[OK] Visualizations complete!")
print("\nKey Findings:")
print(f"- Best MAE: {best_mae['name']} ({best_mae['mae']:.2f} likes)")
print(f"- Best R²: {best_r2['name']} ({best_r2['r2']:.4f})")

# Count good vs bad features
good_features = len(comparison_df[comparison_df['mae_improvement'] > 0])
bad_features = len(comparison_df[comparison_df['mae_improvement'] < 0])
print(f"\n- Features that HELP: {good_features}/{len(comparison_df)}")
print(f"- Features that HURT: {bad_features}/{len(comparison_df)}")

# Face detection specific
face_result = comparison_df[comparison_df['group'] == 'face'].iloc[0]
print(f"\n- Face Detection Impact: {face_result['mae_improvement']:+.1f}% MAE, {face_result['r2_improvement']:+.1f}% R²")
if face_result['mae_improvement'] < 0:
    print("  Answer: Face detection DOES NOT help significantly!")
else:
    print("  Answer: Face detection HELPS!")

# Don't show interactive plot, just save
print("\n[OK] All visualizations saved!")
