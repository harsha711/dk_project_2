#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the unified Gradio app
echo "🚀 Starting Dental AI Platform (Unified Chatbot)..."
echo "📱 Open your browser to: http://localhost:7860"
echo ""
echo "Features:"
echo "  - Tab 1: 🤖 Unified Chatbot (Text + Vision)"
echo "  - Tab 2: 📊 Dataset Explorer"
echo ""
python dental_ai_unified.py
