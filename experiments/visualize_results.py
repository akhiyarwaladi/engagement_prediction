#!/usr/bin/env python3
"""
Visualization Dashboard for Experiment Results
Create comprehensive visualizations of all experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_experiment_results():
    """Load all experiment results"""
    results_file = 'experiments/results.jsonl'

    if not Path(results_file).exists():
        print(f"[ERROR] No results found at {results_file}")
        return None

    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"[INFO] Loaded {len(results)} experiments")
    return results


def create_performance_comparison(results, output_dir='docs/figures'):
    """Create performance comparison plots"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract metrics
    data = []
    for r in results:
        data.append({
            'experiment': r['experiment_id'][:30],  # Truncate name
            'name': r['config']['name'],
            'test_mae': r['metrics']['test_mae'],
            'test_r2': r['metrics']['test_r2'],
            'train_mae': r['metrics']['train_mae'],
            'train_r2': r['metrics']['train_r2'],
            'features': len(r['config']['features']),
            'tags': ','.join(r['config'].get('tags', []))
        })

    df = pd.DataFrame(data)

    # 1. MAE Comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df['train_mae'], width, label='Train MAE', alpha=0.7, color='skyblue')
    ax.bar(x + width/2, df['test_mae'], width, label='Test MAE', alpha=0.7, color='coral')

    ax.set_xlabel('Experiment')
    ax.set_ylabel('MAE (likes)')
    ax.set_title('Model Performance: Train vs Test MAE')
    ax.set_xticks(x)
    ax.set_xticklabels(df['name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mae_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/mae_comparison.png")
    plt.close()

    # 2. R² Comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(x - width/2, df['train_r2'], width, label='Train R²', alpha=0.7, color='lightgreen')
    ax.bar(x + width/2, df['test_r2'], width, label='Test R²', alpha=0.7, color='salmon')

    ax.set_xlabel('Experiment')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance: Train vs Test R²')
    ax.set_xticks(x)
    ax.set_xticklabels(df['name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/r2_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/r2_comparison.png")
    plt.close()

    # 3. Overfitting Analysis
    df['overfit_gap'] = abs(df['train_r2'] - df['test_r2'])

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['red' if gap > 0.5 else 'orange' if gap > 0.3 else 'green' for gap in df['overfit_gap']]
    ax.bar(x, df['overfit_gap'], color=colors, alpha=0.7)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Overfitting Gap (|Train R² - Test R²|)')
    ax.set_title('Overfitting Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(df['name'], rotation=45, ha='right')
    ax.axhline(y=0.3, color='orange', linestyle='--', label='Warning threshold', alpha=0.5)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Severe threshold', alpha=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/overfitting_analysis.png")
    plt.close()

    # 4. Features vs Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # MAE vs Features
    scatter1 = ax1.scatter(df['features'], df['test_mae'], s=100, alpha=0.6, c=df['overfit_gap'],
                          cmap='RdYlGn_r', edgecolors='black', linewidth=1)
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Test MAE (likes)')
    ax1.set_title('Test MAE vs Number of Features')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Overfit Gap')

    # Annotate best points
    best_mae_idx = df['test_mae'].idxmin()
    ax1.annotate(df.loc[best_mae_idx, 'name'],
                xy=(df.loc[best_mae_idx, 'features'], df.loc[best_mae_idx, 'test_mae']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # R² vs Features
    scatter2 = ax2.scatter(df['features'], df['test_r2'], s=100, alpha=0.6, c=df['overfit_gap'],
                          cmap='RdYlGn_r', edgecolors='black', linewidth=1)
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Test R²')
    ax2.set_title('Test R² vs Number of Features')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Overfit Gap')

    # Annotate best points
    best_r2_idx = df['test_r2'].idxmax()
    ax2.annotate(df.loc[best_r2_idx, 'name'],
                xy=(df.loc[best_r2_idx, 'features'], df.loc[best_r2_idx, 'test_r2']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/features_vs_performance.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/features_vs_performance.png")
    plt.close()

    # 5. Leaderboard Table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_df = df[['name', 'test_mae', 'test_r2', 'features', 'overfit_gap']].copy()
    table_df = table_df.sort_values('test_mae')
    table_df = table_df.head(10)  # Top 10

    table_df.columns = ['Model', 'Test MAE', 'Test R²', 'Features', 'Overfit Gap']
    table_df['Rank'] = range(1, len(table_df) + 1)
    table_df = table_df[['Rank', 'Model', 'Test MAE', 'Test R²', 'Features', 'Overfit Gap']]

    # Format numbers
    table_df['Test MAE'] = table_df['Test MAE'].apply(lambda x: f'{x:.2f}')
    table_df['Test R²'] = table_df['Test R²'].apply(lambda x: f'{x:.3f}')
    table_df['Overfit Gap'] = table_df['Overfit Gap'].apply(lambda x: f'{x:.3f}')

    table = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code
    for i in range(1, len(table_df) + 1):
        if i == 1:
            table[(i, 0)].set_facecolor('gold')
        elif i == 2:
            table[(i, 0)].set_facecolor('silver')
        elif i == 3:
            table[(i, 0)].set_facecolor('#CD7F32')  # Bronze

    plt.title('Top 10 Models Leaderboard (by Test MAE)', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/leaderboard.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/leaderboard.png")
    plt.close()

    return df


def create_summary_report(results, output_file='experiments/EXPERIMENT_SUMMARY.md'):
    """Create markdown summary report"""

    # Extract data
    data = []
    for r in results:
        data.append({
            'name': r['config']['name'],
            'description': r['config']['description'],
            'test_mae': r['metrics']['test_mae'],
            'test_r2': r['metrics']['test_r2'],
            'train_mae': r['metrics']['train_mae'],
            'train_r2': r['metrics']['train_r2'],
            'features': len(r['config']['features']),
            'duration': r['duration_seconds'],
            'timestamp': r['timestamp']
        })

    df = pd.DataFrame(data)

    # Create report
    with open(output_file, 'w') as f:
        f.write("# EXPERIMENT SUMMARY REPORT\n\n")
        f.write(f"**Total Experiments:** {len(results)}\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n")

        f.write("## BEST MODELS\n\n")

        # Best MAE
        best_mae = df.loc[df['test_mae'].idxmin()]
        f.write(f"### Best Test MAE: {best_mae['name']}\n\n")
        f.write(f"- **Test MAE:** {best_mae['test_mae']:.2f} likes\n")
        f.write(f"- **Test R²:** {best_mae['test_r2']:.3f}\n")
        f.write(f"- **Features:** {best_mae['features']}\n")
        f.write(f"- **Description:** {best_mae['description']}\n\n")

        # Best R²
        best_r2 = df.loc[df['test_r2'].idxmax()]
        f.write(f"### Best Test R²: {best_r2['name']}\n\n")
        f.write(f"- **Test R²:** {best_r2['test_r2']:.3f}\n")
        f.write(f"- **Test MAE:** {best_r2['test_mae']:.2f} likes\n")
        f.write(f"- **Features:** {best_r2['features']}\n")
        f.write(f"- **Description:** {best_r2['description']}\n\n")

        f.write("---\n\n")

        f.write("## ALL EXPERIMENTS\n\n")
        f.write("| Rank | Model | Test MAE | Test R² | Features | Overfit Gap |\n")
        f.write("|------|-------|----------|---------|----------|-------------|\n")

        df_sorted = df.sort_values('test_mae')
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            overfit = abs(row['train_r2'] - row['test_r2'])
            f.write(f"| {i} | {row['name']} | {row['test_mae']:.2f} | {row['test_r2']:.3f} | {row['features']} | {overfit:.3f} |\n")

        f.write("\n")

    print(f"[SAVED] Summary report: {output_file}")


def main():
    """Main execution"""
    print("=" * 80)
    print("EXPERIMENT VISUALIZATION DASHBOARD")
    print("=" * 80)

    # Load results
    results = load_experiment_results()

    if not results:
        print("[ERROR] No results to visualize")
        return

    # Create visualizations
    print("\n[VIZ] Creating visualizations...")
    df = create_performance_comparison(results)

    # Create summary report
    print("\n[REPORT] Generating summary report...")
    create_summary_report(results)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - docs/figures/mae_comparison.png")
    print("  - docs/figures/r2_comparison.png")
    print("  - docs/figures/overfitting_analysis.png")
    print("  - docs/figures/features_vs_performance.png")
    print("  - docs/figures/leaderboard.png")
    print("  - experiments/EXPERIMENT_SUMMARY.md")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
