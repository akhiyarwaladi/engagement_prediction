#!/usr/bin/env python3
"""
Main script to run the complete ML pipeline.

This script will:
1. Extract features from raw Instagram data
2. Train Random Forest model
3. Evaluate performance
4. Save model and visualizations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.trainer import ModelTrainer
from src.utils import load_config, setup_logger


def main():
    """Run the complete ML pipeline."""

    print("\n" + "=" * 80)
    print(" " * 20 + "INSTAGRAM ENGAGEMENT PREDICTION")
    print(" " * 25 + "FST UNJA Research Project")
    print("=" * 80)

    # Load config
    config = load_config()

    # Setup logger
    logger = setup_logger(
        'Main',
        log_file=config['logging']['file']
    )

    logger.info("Starting Instagram Engagement Prediction Pipeline")

    try:
        # Initialize trainer
        trainer = ModelTrainer()

        # Run complete training pipeline
        metrics_df = trainer.run()

        # Print summary
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUCCESSFUL")
        print("=" * 80)

        print("\n📊 Final Evaluation Metrics:")
        print("-" * 80)
        for col in metrics_df.columns:
            value = metrics_df[col].values[0]
            print(f"  {col:20s}: {value:.4f}")
        print("-" * 80)

        print("\n✅ Model saved to: models/baseline_rf_model.pkl")
        print("✅ Plots saved to: docs/figures/")
        print("✅ Logs saved to: logs/training.log")

        # Feature importance
        importance_df = trainer.model.get_feature_importance()
        print("\n📈 Top 5 Most Important Features:")
        print("-" * 80)
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {idx+1}. {row['feature']:20s}: {row['importance']:.4f}")
        print("-" * 80)

        # Performance assessment
        mae_test = metrics_df['MAE_test'].values[0]
        r2_test = metrics_df['R2_test'].values[0]

        target_mae = config['evaluation']['target_performance']['mae_max']
        target_r2 = config['evaluation']['target_performance']['r2_min']

        print("\n🎯 Performance Assessment:")
        print("-" * 80)

        if mae_test <= target_mae:
            print(f"  ✅ MAE Target: ACHIEVED ({mae_test:.2f} <= {target_mae})")
        else:
            print(f"  ⚠️  MAE Target: NOT MET ({mae_test:.2f} > {target_mae})")

        if r2_test >= target_r2:
            print(f"  ✅ R² Target: ACHIEVED ({r2_test:.3f} >= {target_r2})")
        else:
            print(f"  ⚠️  R² Target: NOT MET ({r2_test:.3f} < {target_r2})")

        print("-" * 80)

        # Next steps
        print("\n📝 Next Steps:")
        print("-" * 80)
        print("  1. Review plots in docs/figures/")
        print("  2. Check logs in logs/training.log for details")
        print("  3. Use predict.py to make predictions on new posts")
        print("  4. (Optional) Run Streamlit app: streamlit run app/streamlit_app.py")
        print("-" * 80)

        print("\n" + "=" * 80)
        print("DONE! 🎉")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        print(f"\n❌ ERROR: {str(e)}")
        print("Check logs/training.log for details")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
