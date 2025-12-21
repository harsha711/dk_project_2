# 📚 Quick Reference Guide

## 🚀 Quick Start

### Run the Application
```bash
cd backend
./run_unified.sh
# Or: python dental_ai_unified.py
```

### Setup Environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure API Keys
Create `backend/.env`:
```env
OPEN_AI_API_KEY=your_key_here
GROQ_AI_API_KEY=your_key_here
```

---

## 📁 File Structure

```
backend/
├── dental_ai_unified.py      # Main app
├── api_utils.py               # YOLO + API calls
├── multimodal_utils.py        # Routing & formatting
├── image_utils.py             # Image processing
├── dataset_utils.py           # Dataset manager
├── models/
│   └── dental_impacted.pt     # Trained YOLO model
└── run_unified.sh             # Launcher
```

---

## 🎯 Key Functions

### YOLO Detection
```python
from api_utils import detect_teeth_yolo
result = detect_teeth_yolo(image, conf_threshold=0.35)
```

### Multi-Model Chat
```python
from api_utils import multimodal_chat_async
responses = await multimodal_chat_async(
    message, image, context, models, clients
)
```

### Image Annotation
```python
from image_utils import draw_bounding_boxes
annotated = draw_bounding_boxes(image, detections)
```

---

## 🎓 Training Quick Commands

### Train Model
```bash
python train_yolo_dental.py --epochs 100 --batch 16
```

### Check Training
```bash
./check_training.sh
```

### Inspect Dataset
```bash
python inspect_dataset.py --data Dental-X-ray-1/data.yaml
```

---

## 🔧 Configuration

### Model Path
```python
# In api_utils.py
YOLO_IMPACTED_MODEL_PATH = "models/dental_impacted.pt"
```

### Confidence Thresholds
```python
# In api_utils.py
CLASS_CONFIDENCE_THRESHOLDS = {
    "Impacted": 0.25,
    "Cavity": 0.30,
    # ...
}
```

---

## 📊 Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | yolov8n.pt | YOLOv8 Nano |
| Epochs | 100 | Max training epochs |
| Batch | 16 | Batch size |
| Image Size | 640 | 640x640 pixels |
| Learning Rate | 0.01 | Initial LR |
| Patience | 20 | Early stopping |

---

## 🎨 Model Classes

- **Impacted**: Impacted wisdom teeth
- **Cavity**: Dental cavities
- **Deep Caries**: Deep tooth decay
- **Filling**: Dental fillings
- **Implant**: Dental implants
- **Crown**: Dental crowns

---

## 🔍 Troubleshooting

### Model Not Found
```bash
# Download base model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### API Errors
- Check `.env` file exists
- Verify API keys are correct
- Check API quotas

### Memory Issues
- Reduce batch size
- Use smaller model (nano)
- Reduce image size

---

## 📖 Documentation Links

- **[Complete Documentation](COMPLETE_DOCUMENTATION.md)** - Full project docs
- **[Training Guide](TRAINING_GUIDE.md)** - Detailed training docs
- **[Architecture](ARCHITECTURE.md)** - System architecture
- **[Quick Start](QUICKSTART.md)** - Getting started guide

---

*Last Updated: 2025*

