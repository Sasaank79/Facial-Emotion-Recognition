#!/bin/bash
# Safe demo runner with virtual environment

echo "🎭 Emotion Detection Demo - Safe Runner"
echo "========================================"

# Create venv if it doesn't exist
if [ ! -d "demo_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv demo_venv
    
    echo "Installing dependencies..."
    source demo_venv/bin/activate
    pip install --quiet torch torchvision timm albumentations opencv-python-headless "numpy<2"
    pip install --quiet 'protobuf<5,>=4.25.3' absl-py 'attrs>=19.1.0' 'flatbuffers>=2.0' --no-deps
    pip install --quiet mediapipe --no-deps
    deactivate
    echo "✅ Virtual environment created!"
else
    echo "✅ Using existing virtual environment"
fi

# Activate and run
echo ""
echo "Starting demo..."
source demo_venv/bin/activate
python demo_webcam_fixed.py --debug
deactivate

echo ""
echo "✅ Demo closed - your system packages are unchanged!"
