#!/bin/bash

# Setup and train YOLO model for dental X-ray detection
# This script sets up the environment and starts training

set -e  # Exit on error

echo "🔧 Setting up YOLO training environment..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install ultralytics roboflow
    echo "✅ Virtual environment created and packages installed"
fi

# Check if dataset exists
if [ ! -f "Dental-X-ray-1/data.yaml" ] && [ ! -f "datasets/Dental-X-ray-1/data.yaml" ]; then
    echo "⚠️  Dataset not found. Please download the dataset first."
    echo "   Run: python download_roboflow_dataset.py"
    exit 1
fi

# Find dataset path
if [ -f "Dental-X-ray-1/data.yaml" ]; then
    DATA_PATH="Dental-X-ray-1/data.yaml"
elif [ -f "datasets/Dental-X-ray-1/data.yaml" ]; then
    DATA_PATH="datasets/Dental-X-ray-1/data.yaml"
fi

echo "📊 Dataset found: $DATA_PATH"
echo "🚀 Starting training..."
echo ""

# Run training with default parameters
python train_yolo_dental.py \
    --model n \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --data "$DATA_PATH" \
    --project dental_detection

echo ""
echo "✅ Training script completed!"
echo "📁 Check the runs/ directory for results"

