# 🦷 Dental AI Platform v2.4

**Multi-Model AI System for Dental X-Ray Analysis**

Combines YOLOv8 object detection with 3 text-based AI models to provide accurate, visual analysis of dental pathologies.

---

## 🎯 Key Features

✅ **Custom YOLO Detection** - Trained on 1,075 dental X-rays (88% mAP accuracy)  
✅ **Multi-Model AI Analysis** - GPT-4o-mini, Llama 3.3 70B, Qwen 3 32B (consensus)  
✅ **Visual Annotations** - Automatic bounding boxes on detected pathologies  
✅ **Conversational Interface** - Ask follow-up questions about findings  
✅ **Real-Time Processing** - <100ms detection + parallel AI inference  
✅ **Smart Dataset Filtering** - Automatically skips binary/mask images, only shows real X-rays  
✅ **Local Dataset Support** - Automatically detects and uses local YOLO format datasets

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB RAM minimum
- GPU recommended (but not required)

### Installation

```bash
# 1. Navigate to backend
cd backend

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cat > .env << EOF
OPEN_AI_API_KEY=your_openai_key
GROQ_AI_API_KEY=your_groq_key
EOF

# 5. Launch application
./run_unified.sh
```

Access at: **http://localhost:7860**

---

## 📚 Documentation

**📖 [DENTAL_AI_COMPLETE_GUIDE.md](DENTAL_AI_COMPLETE_GUIDE.md)** - Complete consolidated documentation covering:
- Quick start guide
- System architecture
- Installation & setup
- Usage guide
- YOLO training guide
- API reference
- Configuration
- Troubleshooting
- Project evolution
- Quick reference

This is the **single source of truth** for all project documentation.

---

## 🏗️ System Architecture

```
User Interface (Gradio Web App)
        ↓
Message Routing & Processing
        ↓
   ┌────────┴─────────┐
   ↓                  ↓
YOLO Detection    Text AI Models
(Custom Model)    (3 in Parallel)
   ↓                  ↓
Bounding Boxes   Clinical Analysis
   ↓                  ↓
   └────────┬─────────┘
            ↓
   Formatted Response
```

---

## 📊 Model Performance

**Custom YOLOv8 Model**:
- Overall mAP@50: **88.0%**
- Impacted Tooth Precision: **82.2%**
- Inference Speed: **~5ms per image**
- Training Time: **9 minutes** (GPU)

**Dataset**: 1,075 annotated dental X-rays from Roboflow

---

## 🛠️ Technology Stack

- **Object Detection**: YOLOv8n (Ultralytics)
- **AI Models**: GPT-4o-mini, Llama 3.3 70B, Qwen 3 32B
- **Web Interface**: Gradio
- **Deep Learning**: PyTorch 2.9.1 + CUDA 12.8
- **Image Processing**: Pillow

---

## 📁 Project Structure

```
dk_project_2/
├── backend/
│   ├── models/dental_impacted.pt      # Trained YOLO model
│   ├── dental_ai_unified.py           # Main application
│   ├── api_utils.py                   # AI integrations + YOLO
│   ├── multimodal_utils.py            # Message routing
│   ├── image_utils.py                 # Image processing
│   ├── train_yolo_dental.py           # Training script
│   └── run_unified.sh                 # Launch script
├── docs/
│   ├── COMPLETE_DOCUMENTATION.md
│   └── TRAINING_GUIDE.md
├── README.md
├── COMPLETE_DOCUMENTATION.md
└── PROJECT_EVOLUTION.md
```

---

## 🎓 What It Detects

| Pathology | Confidence Threshold | Spatial Filtering |
|-----------|---------------------|-------------------|
| Impacted Tooth | 0.25 | Jaw edges only (x<0.30 or x>0.70) |
| Cavity | 0.30 | None |
| Fillings | 0.30 | None |
| Implant | 0.30 | None |

---

## ⚙️ Configuration

### Adjust Detection Sensitivity

Edit `backend/api_utils.py` (lines 71-83):
```python
CLASS_CONFIDENCE_THRESHOLDS = {
    "Impacted Tooth": 0.25,  # Lower = more sensitive
    ...
}
```

### Change AI Models

Edit `backend/multimodal_utils.py` (line 118):
```python
mode, models = "vision", ["gpt4", "groq", "qwen"]
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python train_yolo_dental.py` |
| API key errors | Check `backend/.env` file |
| CUDA out of memory | Reduce batch size in training script |
| Port in use | Change port in `dental_ai_unified.py` |

See [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#troubleshooting) for details.

---

## ⚠️ Disclaimer

**For educational and research purposes only.**  
NOT for clinical diagnosis. Consult qualified dental professionals.

---

## 📜 License

- YOLOv8: AGPL-3.0 (Ultralytics)
- Dataset: CC BY 4.0 (Roboflow)
- AI APIs: See provider terms

---

## 👨‍💻 Author

**Harsha**  
Version: 2.4  
December 2025

---

## 🙏 Acknowledgments

- Ultralytics (YOLOv8)
- Roboflow (Dataset)
- Groq (Llama & Qwen inference)
- OpenAI (GPT-4o-mini)

---


