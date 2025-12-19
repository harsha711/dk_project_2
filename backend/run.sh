#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the Gradio app
echo "🚀 Starting Dental AI Platform..."
echo "📱 Open your browser to: http://localhost:7860"
echo ""
python dental_ai_app.py
