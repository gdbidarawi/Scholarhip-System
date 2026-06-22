#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  ScholarAI — Launch Script
#  Usage:  bash run.sh
# ─────────────────────────────────────────────────────────────────
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         ScholarAI — Scholarship ML System            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# 1. Install dependencies
echo "📦 Installing dependencies…"
pip install -r requirements.txt --break-system-packages -q

# 2. Train model (if not already trained)
cd backend
if [ ! -f "model/scholarship_model.pkl" ]; then
  echo "🧠 Training ML model (first run)…"
  python3 train_model.py
else
  echo "✅ Pre-trained model found — skipping training."
fi

echo ""
echo "🚀 Starting Flask API on http://localhost:5050"
echo "🌐 Open frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop."
echo ""

python3 app.py
