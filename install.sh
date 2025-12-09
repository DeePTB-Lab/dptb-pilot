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

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    pip install uv
fi

# Sync dependencies with the specified find-links
echo "Installing dependencies..."
# We use uv sync to install the project and dependencies
uv sync --find-links "$FIND_LINKS_URL"

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
echo "alias pilot='uv run --project $PROJECT_ROOT dptb-pilot'"
echo "alias tools='uv run --project $PROJECT_ROOT dptb-tools'"

echo ""
