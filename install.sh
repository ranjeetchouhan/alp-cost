#!/usr/bin/env bash
set -e

echo "=============================================="
echo "⚡ Installing Alp-Cost: AI Cost-Shrinker"
echo "=============================================="

# 1. Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✔ Found Python $PYTHON_VERSION"

# 2. Set up virtual environment
echo "✔ Creating virtual environment (.venv)..."
python3 -m venv .venv

# 3. Install dependencies
echo "✔ Installing dependencies from requirements.txt..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 4. Create default .env if not exists
if [ ! -f .env ]; then
    echo "✔ Creating default .env configuration..."
    cat << 'ENVEOF' > .env
UPSTREAM_BASE_URL="http://localhost:11434/v1"
UPSTREAM_API_KEY="ollama"
SEMANTIC_THRESHOLD="0.88"
ENVEOF
fi

# 5. Create data directory
mkdir -p data

echo "=============================================="
echo "✅ Alp-Cost Installation Complete!"
echo "To start the server, run:"
echo "    .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080"
echo "=============================================="
