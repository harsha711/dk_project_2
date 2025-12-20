#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the enhanced Gradio app with dataset features
echo "🚀 Starting Enhanced Dental AI Platform..."
echo "📊 With Hugging Face Dataset Integration (1,206 samples)"
echo "📱 Open your browser to: http://localhost:7860"
echo ""
echo "Features:"
echo "  - Tab 1: Wisdom Tooth Detection"
echo "  - Tab 2: Multi-Model Chatbot"
echo "  - Tab 3: Dataset Explorer (NEW!)"
echo ""
python dental_ai_enhanced.py
