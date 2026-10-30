#!/bin/bash
# dptb-pilot Installation Script
#
# Usage:
#   ./install.sh           # Install CPU version (default)
#   ./install.sh cpu       # Install CPU version
#   ./install.sh cu118     # Install CUDA 11.8 version
#   ./install.sh cu121     # Install CUDA 12.1 version
#   ./install.sh cu124     # Install CUDA 12.4 version

set -e  # Exit on error

# Default to CPU
VARIANT="${1:-cpu}"

# Validate variant
case "$VARIANT" in
    cpu|cu118|cu121|cu124)
        ;;
    *)
        echo "❌ Invalid variant: $VARIANT"
        echo "Allowed variants: cpu, cu118, cu121, cu124"
        exit 1
        ;;
esac

# Detect torch version from pyproject.toml (we pin to 2.5.0 for stability with these wheels)
TORCH_VERSION="2.5.0"

# Set the find-links URL based on variant
FIND_LINKS_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${VARIANT}.html"

echo "======================================"
echo "dptb-pilot Installation Script"
echo "======================================"
echo "PyTorch variant: $VARIANT"
echo "Find-links URL: $FIND_LINKS_URL"
echo "======================================"
echo ""

# Check python environment
echo "Using Python: $(which python)"
python - <<'EOF'
import sys
print("Python executable:", sys.executable)
EOF

# Check if uv is installed
if ! python -c "import uv" &>/dev/null; then
    echo "UV is not installed in this Python environment. Installing..."
    python -m pip install --upgrade pip
    python -m pip install uv
fi


# Sync dependencies with the specified find-links
echo "Installing dependencies..."
# We use uv sync to install the project and dependencies
# We use uv sync to install the project and dependencies
if ! python -m uv sync --find-links "$FIND_LINKS_URL"; then
    echo ""
    echo "⚠️  Installation failed using the existing lockfile."
    echo "This often happens when installing on a different OS (e.g., Linux vs Mac)."
    echo "🔄 Backing up uv.lock and retrying with a fresh resolution..."
    
    mv uv.lock uv.lock.bak 2>/dev/null || true
    
    if ! python -m uv sync --find-links "$FIND_LINKS_URL"; then
        echo "❌ Installation failed even after refreshing lockfile."
        exit 1
    fi
fi

# Copy offline embedding model
echo "Copy offline embedding model..."
cp -r resources/emb_model/huggingface ~/.cache/huggingface/

echo ""
echo "✅ Installation complete!"
echo ""
echo "To run the pilot:"
echo "  uv run dptb-pilot"
echo ""
echo "To run the tools server:"
echo "  uv run dptb-tools"
echo ""

# Frontend Installation
echo "======================================"
echo "Frontend Installation"
echo "======================================"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js and npm to build the frontend."
    echo "Skipping frontend installation."
else
    echo "Installing frontend dependencies..."
    if [ -d "web_ui" ]; then
        cd web_ui
        npm install
        
        echo "Building frontend..."
        npm run build
        cd ..
        echo "✅ Frontend installation complete!"
    else
        echo "❌ web_ui directory not found!"
    fi
fi

echo ""
echo "🎉 All Done! You are ready to go."
echo ""

echo "======================================"
echo "🌍 Run dptb-pilot from ANYWHERE"
echo "======================================"

PROJECT_ROOT=$(pwd)

echo "Add these aliases to your shell config (e.g., ~/.zshrc or ~/.bashrc):"
echo ""
echo "# One-click startup (Recommended)"
echo "alias dptb-ai-run=$PROJECT_ROOT/start.sh"
echo ""
echo "# Individual components"
echo "alias dptb-pilot='uv run --project $PROJECT_ROOT dptb-pilot'"
echo "alias dptb-tools='uv run --project $PROJECT_ROOT dptb-tools'"

echo ""
echo "Or copy/paste them into your current terminal to use them immediately!"
echo ""
