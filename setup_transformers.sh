#!/bin/bash
# Setup script for transformer-based models
# Phase 4: IndoBERT + ViT + CLIP implementation

echo "=================================="
echo "PHASE 4 TRANSFORMER SETUP"
echo "=================================="
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in virtual environment!"
    echo "   Activating venv..."
    source venv/bin/activate
fi

echo "📦 Installing PyTorch and Transformers..."
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA detected! Installing PyTorch with GPU support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    echo "💻 No CUDA detected. Installing PyTorch CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi

echo ""
echo "📚 Installing Transformers library..."
pip install transformers --quiet

echo ""
echo "📊 Installing additional dependencies..."
pip install sentencepiece protobuf --quiet

echo ""
echo "🔍 Testing installation..."
python3 << 'EOF'
import torch
import transformers

print("✅ PyTorch version:", torch.__version__)
print("✅ Transformers version:", transformers.__version__)

# Check if CUDA available
if torch.cuda.is_available():
    print("✅ CUDA available:", torch.cuda.get_device_name(0))
else:
    print("ℹ️  CUDA not available, using CPU (slower but works)")

print("")
print("📥 Downloading IndoBERTweet model (one-time, ~500MB)...")
print("   This may take 2-5 minutes...")

from transformers import AutoTokenizer, AutoModel

try:
    tokenizer = AutoTokenizer.from_pretrained("indolem/IndoBERTweet")
    model = AutoModel.from_pretrained("indolem/IndoBERTweet")
    print("✅ IndoBERTweet model downloaded successfully!")
except Exception as e:
    print("⚠️  Error downloading model:", e)
    print("   Trying alternative: indobenchmark/indobert-base-p1...")
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
    print("✅ IndoBERT model downloaded successfully!")

print("")
print("🧪 Testing model on sample caption...")

sample = "Selamat datang mahasiswa baru FST UNJA! 🎓"
inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=128)

import torch.nn.functional as F
with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()

print(f"   Input: {sample}")
print(f"   Embedding shape: {embedding.shape}")
print(f"   ✅ Model working correctly!")

EOF

echo ""
echo "=================================="
echo "✅ SETUP COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run: python3 extract_bert_features.py"
echo "2. Run: python3 improve_model_v4_bert.py"
echo "3. Expected: MAE ~95, R² ~0.25"
echo ""
echo "For visual features (ViT):"
echo "4. Run: python3 extract_vit_features.py"
echo "5. Run: python3 improve_model_v4_full.py"
echo "6. Expected: MAE ~75, R² ~0.35"
echo ""
