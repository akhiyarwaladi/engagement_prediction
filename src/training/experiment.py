#!/usr/bin/env python3
"""
Experiment tracking and management system
Tracks all experiments with results, config, and artifacts
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str
    description: str
    features: List[str]
    model_type: str
    model_params: Dict[str, Any]
    preprocessing: Dict[str, Any]
    random_state: int = 42
    tags: List[str] = None

    def to_dict(self):
        return asdict(self)

    def get_hash(self):
        """Generate unique hash for this config"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResult:
    """Experiment results"""
    experiment_id: str
    config: ExperimentConfig
    metrics: Dict[str, float]
    timestamp: str
    duration_seconds: float
    model_path: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[pd.DataFrame] = None

    def to_dict(self):
        result = {
            'experiment_id': self.experiment_id,
            'config': self.config.to_dict(),
            'metrics': self.metrics,
            'timestamp': self.timestamp,
            'duration_seconds': self.duration_seconds,
            'model_path': self.model_path,
        }
        if self.feature_importance:
            result['feature_importance_top10'] = dict(
                sorted(self.feature_importance.items(),
                      key=lambda x: x[1], reverse=True)[:10]
            )
        return result


class ExperimentTracker:
    """Track and manage experiments"""

    def __init__(self, experiments_dir: str = 'experiments'):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.experiments_dir / 'results.jsonl'
        self.leaderboard_file = self.experiments_dir / 'leaderboard.csv'

    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start new experiment, return experiment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = config.get_hash()
        experiment_id = f"{config.name}_{timestamp}_{config_hash}"

        print(f"\n[EXPERIMENT] Starting: {experiment_id}")
        print(f"  Description: {config.description}")
        if config.tags:
            print(f"  Tags: {', '.join(config.tags)}")

        return experiment_id

    def log_result(self, result: ExperimentResult):
        """Log experiment result"""

        # Append to results file (JSONL format)
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')

        # Update leaderboard
        self._update_leaderboard()

        print(f"\n[RESULT] Experiment: {result.experiment_id}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        for metric, value in result.metrics.items():
            print(f"  {metric}: {value:.4f}")

        if result.model_path:
            print(f"  Model saved: {result.model_path}")

    def _update_leaderboard(self):
        """Update leaderboard CSV"""

        # Read all results
        results = []
        if self.results_file.exists():
            with open(self.results_file) as f:
                for line in f:
                    results.append(json.loads(line))

        if not results:
            return

        # Create leaderboard DataFrame
        rows = []
        for r in results:
            row = {
                'experiment_id': r['experiment_id'],
                'name': r['config']['name'],
                'timestamp': r['timestamp'],
                'duration_s': r['duration_seconds'],
                'model_type': r['config']['model_type'],
            }
            # Add all metrics
            row.update(r['metrics'])

            # Add tags
            tags = r['config'].get('tags', [])
            row['tags'] = ','.join(tags) if tags else ''

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by MAE (best first)
        if 'test_mae' in df.columns:
            df = df.sort_values('test_mae')

        # Save
        df.to_csv(self.leaderboard_file, index=False)

    def get_leaderboard(self, metric='test_mae', top_k=10) -> pd.DataFrame:
        """Get top experiments"""

        if not self.leaderboard_file.exists():
            return pd.DataFrame()

        df = pd.read_csv(self.leaderboard_file)

        # Sort by metric
        ascending = True if 'mae' in metric.lower() or 'rmse' in metric.lower() else False
        df = df.sort_values(metric, ascending=ascending)

        return df.head(top_k)

    def get_best_experiment(self, metric='test_mae') -> Optional[Dict]:
        """Get best experiment by metric"""

        if not self.results_file.exists():
            return None

        results = []
        with open(self.results_file) as f:
            for line in f:
                results.append(json.loads(line))

        if not results:
            return None

        # Find best
        ascending = True if 'mae' in metric.lower() or 'rmse' in metric.lower() else False
        best = sorted(results, key=lambda x: x['metrics'][metric],
                     reverse=not ascending)[0]

        return best

    def search_experiments(self, tags: Optional[List[str]] = None,
                          name_pattern: Optional[str] = None) -> List[Dict]:
        """Search experiments by tags or name"""

        if not self.results_file.exists():
            return []

        results = []
        with open(self.results_file) as f:
            for line in f:
                result = json.loads(line)

                # Filter by tags
                if tags:
                    result_tags = result['config'].get('tags', [])
                    if not any(tag in result_tags for tag in tags):
                        continue

                # Filter by name
                if name_pattern:
                    if name_pattern.lower() not in result['config']['name'].lower():
                        continue

                results.append(result)

        return results

    def print_summary(self):
        """Print experiment summary"""

        if not self.leaderboard_file.exists():
            print("[INFO] No experiments yet")
            return

        df = pd.read_csv(self.leaderboard_file)

        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        print(f"\nTotal experiments: {len(df)}")

        # Best by MAE
        if 'test_mae' in df.columns:
            best_mae = df.sort_values('test_mae').iloc[0]
            print(f"\n[BEST MAE] {best_mae['experiment_id']}")
            print(f"  MAE: {best_mae['test_mae']:.2f}")
            if 'test_r2' in df.columns:
                print(f"  R²: {best_mae['test_r2']:.3f}")

        # Best by R²
        if 'test_r2' in df.columns:
            best_r2 = df.sort_values('test_r2', ascending=False).iloc[0]
            print(f"\n[BEST R²] {best_r2['experiment_id']}")
            print(f"  R²: {best_r2['test_r2']:.3f}")
            if 'test_mae' in df.columns:
                print(f"  MAE: {best_r2['test_mae']:.2f}")

        # Recent experiments
        print(f"\n[RECENT] Last 5 experiments:")
        recent = df.tail(5)
        for _, row in recent.iterrows():
            mae = row.get('test_mae', -1)
            r2 = row.get('test_r2', -1)
            print(f"  {row['experiment_id'][:30]:30s}  MAE={mae:6.2f}  R²={r2:6.3f}")

    def export_results(self, output_file='experiment_results.json'):
        """Export all results to JSON"""

        if not self.results_file.exists():
            print("[INFO] No results to export")
            return

        results = []
        with open(self.results_file) as f:
            for line in f:
                results.append(json.loads(line))

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[EXPORT] Exported {len(results)} results to {output_file}")


# Global tracker instance
_tracker = None


def get_tracker(experiments_dir='experiments') -> ExperimentTracker:
    """Get global experiment tracker"""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker(experiments_dir)
    return _tracker
