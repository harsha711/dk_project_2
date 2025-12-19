#!/bin/bash

echo "🚀 Setting up Dental AI Platform"
echo "======================================"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Install pip if not available
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "📦 Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the Dental AI Platform:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the app: python dental_ai_app.py"
echo "  3. Open your browser to: http://localhost:7860"
echo ""
echo "OR simply run: ./run.sh"
echo ""
