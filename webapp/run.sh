#!/bin/bash
# ══════════════════════════════════════════════════════════
#  PhytoGuard AI — Single-Command Launcher
#  Usage: bash run.sh
#  Or:    chmod +x run.sh && ./run.sh
# ══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_PATH="$PROJECT_ROOT/m4_tensorflow"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          PhytoGuard AI — Plant Disease Detection     ║"
echo "║          Vision Transformer · PyTorch · Flask        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
if [ -d "$ENV_PATH" ]; then
  echo "→ Activating m4_tensorflow environment..."
  source "$ENV_PATH/bin/activate"
else
  echo "⚠  m4_tensorflow environment not found at: $ENV_PATH"
  echo "   Attempting to use current Python environment..."
fi

# Install Flask if needed
echo "→ Checking Flask installation..."
python -c "import flask" 2>/dev/null || {
  echo "→ Installing Flask..."
  pip install flask --quiet
}

# Move to webapp directory and start server
cd "$SCRIPT_DIR"
echo "→ Starting PhytoGuard AI server..."
echo "→ Open your browser at: http://localhost:5001"
echo ""

python app.py
