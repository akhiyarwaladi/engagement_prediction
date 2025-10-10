#!/bin/bash
# Ultrathink Experiment Monitoring Script
# Monitor all Phase 11 background experiments

echo "========================================="
echo "ULTRATHINK EXPERIMENT MONITOR - PHASE 11"
echo "========================================="
echo ""

# Check Python processes
echo "[1] Active Python Processes:"
ps aux | grep "phase11" | grep -v grep | wc -l
echo ""

# Monitor Phase 11.3 (PCA Fine-tuning)
echo "[2] Phase 11.3 - PCA Fine-tuning (62-68 components):"
if [ -f "phase11_3_pca_finetune_log.txt" ]; then
    echo "   Status: RUNNING"
    echo "   Latest output:"
    tail -3 phase11_3_pca_finetune_log.txt
else
    echo "   Status: NOT STARTED or NO LOG"
fi
echo ""

# Monitor Phase 11.4 (Emoji + Sentiment)
echo "[3] Phase 11.4 - Emoji + Sentiment Features:"
if [ -f "phase11_4_emoji_sentiment_log.txt" ]; then
    echo "   Status: RUNNING"
    echo "   Latest output:"
    tail -3 phase11_4_emoji_sentiment_log.txt
else
    echo "   Status: NOT STARTED or NO LOG"
fi
echo ""

# Check for completed models
echo "[4] Completed Models (Phase 11):"
ls -lh models/phase11*.pkl 2>/dev/null | wc -l
echo ""

# Extracted features
echo "[5] Extracted Features:"
echo "   - BERT embeddings: $([ -f 'data/processed/bert_embeddings_multi_account.csv' ] && echo 'OK' || echo 'MISSING')"
echo "   - Visual features: $([ -f 'data/processed/advanced_visual_features_multi_account.csv' ] && echo 'OK' || echo 'MISSING')"
echo "   - Emoji/Sentiment: $([ -f 'data/processed/emoji_sentiment_features_multi_account.csv' ] && echo 'OK' || echo 'MISSING')"
echo ""

# Summary
echo "[SUMMARY] Phase 11 Progress:"
echo "   Phase 11.1: COMPLETED - MAE=31.76"
echo "   Phase 11.2: COMPLETED - MAE=28.13 (CHAMPION)"
echo "   Phase 11.3: $([ -f 'phase11_3_pca_finetune_log.txt' ] && echo 'RUNNING' || echo 'PENDING')"
echo "   Phase 11.4: $([ -f 'phase11_4_emoji_sentiment_log.txt' ] && echo 'RUNNING' || echo 'PENDING')"
echo ""
echo "Monitor updated: $(date)"
