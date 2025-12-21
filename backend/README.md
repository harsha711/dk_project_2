# Dental AI - Wisdom Tooth Detection

YOLOv8-based dental X-ray analysis with LLM medical descriptions.

## Features

- ✅ Impacted wisdom tooth detection (YOLO)
- ✅ Accurate bounding boxes with confidence scores
- ✅ Medical analysis from LLM vision models
- ✅ Multi-class detection: Impacted, Caries, Deep Caries, Periapical Lesion

## Quick Start

```bash
# Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Run application
python dental_ai_unified.py
```

Open: http://localhost:7860

## Model

- **DENTEX YOLOv8** (5.9 MB)
- Trained on 1,005 panoramic X-rays
- Classes: Impacted, Caries, Deep Caries, Periapical Lesion
- Location: `models/dental_impacted.pt`

## Architecture

```
User uploads X-ray
    ↓
[YOLO] Accurate bounding box detection
    ↓
[LLM] Medical descriptions & analysis
    ↓
[Output] Annotated X-ray + Expert insights
```

## Files

- `dental_ai_unified.py` - Main Gradio app
- `api_utils.py` - YOLO + LLM integration
- `image_utils.py` - Bounding box rendering
- `multimodal_utils.py` - Message routing
- `dataset_utils.py` - HuggingFace dataset manager

## Requirements

- Python 3.10+
- ultralytics (YOLOv8)
- torch, torchvision
- gradio, PIL, numpy
- openai, groq, google-generativeai
