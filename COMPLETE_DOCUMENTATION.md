# Dental AI Platform - Complete Documentation

**Version**: 2.3
**Last Updated**: December 22, 2025
**Author**: Harsha

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Technology Stack](#technology-stack)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [AI Models](#ai-models)
8. [YOLO Detection System](#yolo-detection-system)
9. [File Structure](#file-structure)
10. [API Reference](#api-reference)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)
13. [Development](#development)

---

## Project Overview

The Dental AI Platform is a **multi-model AI system** for analyzing dental X-rays. It combines:
- **YOLOv8 object detection** for precise localization of dental pathologies
- **3 text-based AI models** (GPT-4o-mini, Llama 3.3 70B, Qwen 2.5 32B) for natural language analysis
- **Interactive web interface** built with Gradio
- **Conversation memory** for contextual follow-up questions

### Key Capabilities

✅ **Detect dental pathologies**:
- Impacted wisdom teeth
- Cavities
- Fillings
- Implants

✅ **Multi-model consensus**: Get opinions from 3 different AI models simultaneously

✅ **Visual annotations**: Automatic bounding boxes on detected features

✅ **Conversational interface**: Ask follow-up questions about detected issues

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Gradio)                  │
│                 dental_ai_unified.py                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Message Routing & Processing                   │
│                 multimodal_utils.py                          │
│  - Route to vision or chat mode                             │
│  - Build conversation context                               │
│  - Format multi-model responses                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│  YOLO Detection     │    │  Text AI Models     │
│   api_utils.py      │    │   api_utils.py      │
│  - YOLOv8 inference │    │  - GPT-4o-mini      │
│  - Bounding boxes   │    │  - Llama 3.3 70B    │
│  - Class filtering  │    │  - Qwen 2.5 32B     │
│  - Spatial rules    │    │  - Async processing │
└─────────────────────┘    └─────────────────────┘
          │                          │
          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Annotated Images    │    │ Text Responses      │
│  image_utils.py     │    │  Markdown formatted │
└─────────────────────┘    └─────────────────────┘
```

### Data Flow

**1. Initial X-ray Upload:**
```
User uploads X-ray → YOLO detects pathologies → Draw bounding boxes
→ Send detections to all 3 text models → Format responses → Display
```

**2. Follow-up Question:**
```
User asks question → Extract conversation context (last 5 turns)
→ Include YOLO results → Send to text models → Display responses
```

---

## Features

### 1. Multi-Model AI Analysis

- **GPT-4o-mini** (OpenAI): Fast, affordable, general-purpose
- **Llama 3.3 70B** (via Groq): Open-source, ultra-fast inference
- **Qwen 2.5 32B** (via Groq): Strong reasoning capabilities, good performance

All models run **in parallel** for speed, providing diverse perspectives.

### 2. YOLOv8 Object Detection

**Custom-trained model** on 1,075 dental X-rays from Roboflow:
- Dataset: `dental-x-ray-1imfs`
- Classes: Cavity, Fillings, Impacted Tooth, Implant
- Performance: 88% mAP@50, 62% mAP@50-95
- Inference speed: ~5ms per image (GPU)

**Smart filtering**:
- Class-specific confidence thresholds
- Spatial validation (impacted teeth must be at jaw edges)
- Non-maximum suppression to reduce overlaps

### 3. Visual Annotations

Automatically draws bounding boxes on detected features:
- Color-coded by class
- Confidence scores displayed
- Position labels (lower-left, upper-right, etc.)

### 4. Conversation Memory

- Stores last 5 conversation turns
- Includes YOLO detection results in context
- Enables follow-up questions like "Tell me more about the impacted tooth"

### 5. Interactive Interface

**Gradio web UI** with:
- Image upload
- Chat interface
- Annotated image gallery
- Sample dental X-rays
- Conversation history

---

## Technology Stack

### Core Technologies

| Component | Technology | Version |
|-----------|-----------|---------|
| **AI Framework** | Ultralytics YOLOv8 | 8.3.240 |
| **Deep Learning** | PyTorch | 2.9.1 |
| **Web Interface** | Gradio | Latest |
| **Image Processing** | Pillow (PIL) | Latest |
| **Async** | asyncio | Python 3.10+ |

### AI APIs

| Provider | Model | API Library |
|----------|-------|-------------|
| OpenAI | GPT-4o-mini | `openai` |
| Groq | Llama 3.3 70B, Qwen 2.5 32B | `groq` |

### Development Tools

- **Python**: 3.10+
- **Virtual Environment**: venv
- **GPU**: CUDA 12.8 (optional, fallback to CPU)

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with CUDA support (optional but recommended)
- Internet connection (for API calls)

### Step 1: Clone Repository

```bash
cd /home/harsha/Documents/dk_project_2
```

### Step 2: Create Virtual Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
```
ultralytics>=8.0.0     # YOLOv8
torch>=2.0.0           # PyTorch
openai                 # GPT-4o-mini
groq                   # Llama 3.3 & Qwen 2.5
gradio                 # Web interface
pillow                 # Image processing
numpy                  # Arrays
roboflow               # Dataset management (optional)
```

### Step 4: Configure API Keys

Create a `.env` file in the `backend/` directory:

```bash
OPEN_AI_API_KEY=your_openai_api_key_here
GROQ_AI_API_KEY=your_groq_api_key_here
```

**Get your API keys**:
- OpenAI: https://platform.openai.com/api-keys
- Groq: https://console.groq.com/keys (free tier available)

### Step 5: Download Pre-trained Model

The custom-trained dental YOLO model is already included at:
```
backend/models/dental_impacted.pt
```

If missing, you can:
1. Train your own (see [YOLO Detection System](#yolo-detection-system))
2. Download from the project repository

### Step 6: Launch Application

```bash
./run_unified.sh
```

Or manually:
```bash
source venv/bin/activate
python dental_ai_unified.py
```

Access the interface at: **http://localhost:7860**

---

## Usage Guide

### Basic Workflow

**1. Start the application**
```bash
cd backend
./run_unified.sh
```

**2. Upload a dental X-ray**
- Click the "Upload X-ray" button
- Select a panoramic dental X-ray image
- Supported formats: JPG, PNG

**3. Get AI analysis**
The system automatically:
- Runs YOLO detection
- Draws bounding boxes
- Analyzes with 3 AI models
- Displays results in 3 columns

**4. Ask follow-up questions**
Examples:
- "What treatment is needed for the impacted tooth?"
- "How serious is the cavity?"
- "What are the risks if left untreated?"

### Sample Images

The application includes sample dental X-rays in `backend/sample_images/`:
- Click any sample to load it
- These demonstrate various pathologies

### Interpreting Results

#### YOLO Detection Box

Shows detected pathologies with:
- **Bounding box**: Red rectangle around detected area
- **Label**: Class name (e.g., "Impacted Tooth")
- **Confidence**: Detection certainty (e.g., 0.82 = 82%)
- **Position**: Location description (e.g., "lower-left")

#### AI Model Responses

Each model provides:
- **Findings**: What pathologies were detected
- **Analysis**: Clinical significance
- **Recommendations**: Suggested next steps

**Compare responses** to get a well-rounded understanding.

---

## AI Models

### 1. GPT-4o-mini (OpenAI)

**Characteristics**:
- General-purpose language model
- Fast response time (~2-3 seconds)
- Good medical knowledge
- Cost: ~$0.15 per 1M input tokens

**Strengths**:
- Clear, structured responses
- Good at explaining complex concepts
- Reliable and consistent

**Limitations**:
- Cannot see images directly (uses YOLO results)
- May be more conservative in recommendations

### 2. Llama 3.3 70B (via Groq)

**Characteristics**:
- Open-source Meta model
- **Ultra-fast** inference via Groq (~500ms)
- 70 billion parameters
- Cost: **FREE** via Groq (14,400 requests/day)

**Strengths**:
- Excellent reasoning capabilities
- Good medical domain knowledge
- Very fast responses
- Completely free

**Limitations**:
- May occasionally be verbose
- Cannot see images directly

### 3. Qwen 2.5 32B (via Groq)

**Characteristics**:
- 32 billion parameters
- Cost: **FREE** via Groq
- Fast inference (~500-800ms)

**Strengths**:
- Strong reasoning capabilities
- Excellent at complex analysis
- Good balance of detail and conciseness
- Free to use

**Limitations**:
- Slightly slower than Llama 3.3
- Cannot see images directly

### Model Comparison Table

| Feature | GPT-4o-mini | Llama 3.3 70B | Qwen 2.5 32B |
|---------|-------------|---------------|--------------|
| **Speed** | Fast | Ultra-fast | Fast |
| **Cost** | Cheap | **Free** | **Free** |
| **Quality** | Very Good | Excellent | Very Good |
| **Medical Knowledge** | ✅ Good | ✅ Good | ✅ Good |
| **Reasoning** | ✅ Strong | ✅ Excellent | ✅ Very Strong |
| **Rate Limit** | 5,000 RPM | 30 RPM (free) | 30 RPM (free) |

### Why 3 Models?

**Consensus & Diversity**: Different models may notice different things or provide unique insights.

**Reliability**: If all 3 agree, high confidence. If they disagree, worth investigating further.

**Redundancy**: If one API is down, you still have 2 others.

---

## YOLO Detection System

### Custom Training

The included model (`dental_impacted.pt`) was trained on:

**Dataset**: Roboflow `dental-x-ray-1imfs`
- 753 training images
- 215 validation images
- 107 test images
- Total: 1,075 annotated dental X-rays

**Training Parameters**:
```python
Model: YOLOv8n (nano - 3M parameters)
Epochs: 100 (early stopping at patience=20)
Batch size: 16
Image size: 640×640
Optimizer: AdamW
Device: NVIDIA GeForce RTX 3070 Ti
Training time: ~9 minutes
```

**Performance Metrics**:
```
Overall:
- Precision: 85.4%
- Recall: 83.4%
- mAP@50: 88.0%
- mAP@50-95: 62.0%

Per-Class (Impacted Tooth):
- Precision: 82.2%
- Recall: 74.5%
- mAP@50: 83.6%
- mAP@50-95: 62.8%
```

### Detection Configuration

Location: `backend/api_utils.py`

**Class-Specific Confidence Thresholds**:
```python
CLASS_CONFIDENCE_THRESHOLDS = {
    "Impacted Tooth": 0.25,  # More sensitive for impacted teeth
    "Cavity": 0.30,
    "Fillings": 0.30,
    "Implant": 0.30,
}
```

**Spatial Filtering for Impacted Teeth**:
```python
# Impacted wisdom teeth must be at jaw edges
# Accept: x < 0.30 (left third) or x > 0.70 (right third)
# Reject: Middle teeth (likely false positives)
```

**NMS Configuration**:
```python
base_conf_threshold = 0.15   # Detect everything initially
iou_threshold = 0.4          # Aggressive overlap suppression
```

### Retraining the Model

If you need to retrain:

**1. Download dataset**:
```bash
python download_roboflow_dataset.py YOUR_ROBOFLOW_API_KEY
```

**2. Train**:
```bash
python train_yolo_dental.py --model n --epochs 100 --batch 16
```

**3. Copy model**:
```bash
cp runs/detect/dental_wisdom_detection/weights/best.pt models/dental_impacted.pt
```

**4. Restart application**:
```bash
./run_unified.sh
```

See `TRAINING_GUIDE.md` for detailed instructions.

---

## File Structure

```
dk_project_2/
├── backend/
│   ├── models/
│   │   └── dental_impacted.pt          # Trained YOLO model (6.3 MB)
│   │
│   ├── sample_images/                  # Sample dental X-rays
│   │   ├── sample1.jpg
│   │   └── ...
│   │
│   ├── venv/                           # Python virtual environment
│   │
│   ├── dental_ai_unified.py            # Main application (Gradio UI)
│   ├── api_utils.py                    # AI model integrations & YOLO
│   ├── multimodal_utils.py             # Message routing & formatting
│   ├── image_utils.py                  # Image processing & annotations
│   ├── dataset_utils.py                # Dataset management (optional)
│   │
│   ├── train_yolo_dental.py            # YOLO training script
│   ├── download_roboflow_dataset.py    # Dataset downloader
│   ├── inspect_dataset.py              # Dataset validation
│   │
│   ├── requirements.txt                # Python dependencies
│   ├── .env                            # API keys (not in git)
│   │
│   ├── run_unified.sh                  # Launch script
│   ├── setup_and_train.sh              # Training setup script
│   └── check_training.sh               # Training progress monitor
│
├── docs/
│   ├── COMPLETE_DOCUMENTATION.md       # This file
│   ├── QUICK_REFERENCE.md              # Quick start guide
│   └── TRAINING_GUIDE.md               # Detailed training guide
│
├── runs/                               # Training outputs (auto-generated)
│   └── detect/
│       └── dental_wisdom_detection/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png             # Training curves
│           ├── confusion_matrix.png
│           └── ...
│
├── Dental-X-ray-1/                    # Downloaded dataset (optional)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── README.md                          # Project overview
└── COMPLETE_DOCUMENTATION.md          # Symlink to docs/
```

### Core Files Explained

**dental_ai_unified.py** (44 KB)
- Main application file
- Gradio web interface
- Message handling
- Image upload/display
- Conversation state management

**api_utils.py** (14 KB)
- YOLO model initialization
- `detect_teeth_yolo()`: Main detection function
- `chat_with_context_async()`: AI model communication
- OpenAI and Groq API integrations

**multimodal_utils.py** (10 KB)
- `route_message()`: Determines vision vs chat mode
- `build_conversation_context()`: Manages conversation history
- `format_multi_model_response()`: Formats UI output

**image_utils.py** (7 KB)
- `draw_bounding_boxes()`: Annotates images with detections
- Image processing utilities
- Color coding by class

---

## API Reference

### Main Functions

#### `detect_teeth_yolo(image, conf_threshold=0.35, iou_threshold=0.4)`

**Description**: Runs YOLOv8 inference on dental X-ray

**Parameters**:
- `image` (PIL.Image): Input dental X-ray
- `conf_threshold` (float): Base confidence threshold (default: 0.35)
- `iou_threshold` (float): NMS IoU threshold (default: 0.4)

**Returns** (Dict):
```python
{
    "success": True,
    "model": "YOLOv8 Dental Detection",
    "teeth_found": [
        {
            "position": "lower-left",
            "bbox": [x1, y1, x2, y2],  # Normalized 0-1
            "confidence": 0.82,
            "class_name": "Impacted Tooth",
            "description": "Impacted Tooth (confidence: 0.82)"
        },
        ...
    ],
    "summary": "Detected 2 teeth/dental features using YOLO"
}
```

**Example**:
```python
from PIL import Image
from api_utils import detect_teeth_yolo

image = Image.open("xray.jpg")
result = detect_teeth_yolo(image)

for detection in result["teeth_found"]:
    print(f"{detection['class_name']} at {detection['position']}: {detection['confidence']:.2%}")
```

---

#### `chat_with_context_async(messages, model_name, openai_client, groq_client)`

**Description**: Sends conversation context to AI model asynchronously

**Parameters**:
- `messages` (list): Conversation history in OpenAI format
- `model_name` (str): One of "gpt4", "groq", "qwen"
- `openai_client`: OpenAI client instance
- `groq_client`: Groq client instance

**Returns** (Dict):
```python
{
    "model": "gpt4",
    "response": "Based on the YOLO detection...",
    "success": True
}
```

**Message Format**:
```python
messages = [
    {"role": "system", "content": "You are a dental AI assistant..."},
    {"role": "user", "content": "What do you see in this X-ray?"},
    {"role": "assistant", "content": "YOLO detected an impacted tooth..."},
    {"role": "user", "content": "Is treatment urgent?"}
]
```

---

#### `draw_bounding_boxes(image, detections)`

**Description**: Draws annotated bounding boxes on image

**Parameters**:
- `image` (PIL.Image): Original X-ray image
- `detections` (list): List of detection dictionaries from YOLO

**Returns**:
- `PIL.Image`: Annotated image with bounding boxes

**Visualization**:
- Red rectangles for bounding boxes
- Labels with class name and confidence
- Position text (upper-left, lower-right, etc.)

---

### Configuration Variables

**Location**: `backend/api_utils.py`

```python
# YOLO Model Path
YOLO_IMPACTED_MODEL_PATH = "models/dental_impacted.pt"

# Class-Specific Thresholds
CLASS_CONFIDENCE_THRESHOLDS = {
    "Impacted Tooth": 0.25,
    "Cavity": 0.30,
    "Fillings": 0.30,
    "Implant": 0.30,
}

# Spatial Filtering (Impacted Teeth)
IMPACTED_X_MIN = 0.30  # Left edge
IMPACTED_X_MAX = 0.70  # Right edge

# NMS Settings
BASE_DETECTION_CONF = 0.15
NMS_IOU_THRESHOLD = 0.4
```

---

## Configuration

### Environment Variables

**File**: `backend/.env`

```bash
# Required
OPEN_AI_API_KEY=sk-proj-...        # OpenAI API key
GROQ_AI_API_KEY=gsk_...            # Groq API key

# Optional
MODEL_PATH=models/dental_impacted.pt  # Custom YOLO model path
DEBUG=False                            # Enable debug logging
```

### Model Selection

**Edit**: `backend/multimodal_utils.py`

```python
# Line 118, 122, 125 - Model arrays
mode, models = "vision", ["gpt4", "groq", "qwen"]

# To remove a model:
mode, models = "vision", ["gpt4", "groq"]  # Removed qwen

# To add a model (requires implementation):
mode, models = "vision", ["gpt4", "groq", "qwen", "new_model"]
```

### YOLO Thresholds

**Edit**: `backend/api_utils.py` (lines 71-83)

```python
CLASS_CONFIDENCE_THRESHOLDS = {
    "Impacted Tooth": 0.25,  # ← Adjust this (lower = more sensitive)
    "Cavity": 0.30,
    ...
}

# Spatial filtering (lines 144-151)
if not (center_x < 0.30 or center_x > 0.70):  # ← Adjust boundaries
    skip_detection = True
```

### UI Customization

**Edit**: `backend/dental_ai_unified.py`

**Title & Description** (lines ~690):
```python
gr.Markdown("# 🦷 Dental AI Analysis Platform")
gr.Markdown("Upload dental X-rays for AI-powered detection...")
```

**Sample Images** (lines ~700):
```python
gr.Examples(
    examples=[
        "sample_images/sample1.jpg",
        "sample_images/sample2.jpg",  # Add more here
    ],
    ...
)
```

**Footer** (line 778):
```python
gr.Markdown("""
**Dental AI Platform v2.3** | Your custom text here
""")
```

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'ultralytics'"

**Cause**: Dependencies not installed

**Solution**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

#### 2. "Model file not found: models/dental_impacted.pt"

**Cause**: YOLO model missing

**Solution**:
```bash
# Option A: Train your own
python train_yolo_dental.py

# Option B: Download pre-trained (if available)
# Place in backend/models/dental_impacted.pt
```

---

#### 3. "CUDA out of memory"

**Cause**: GPU memory insufficient

**Solution**:
```python
# Edit train_yolo_dental.py line 66
batch=8,  # Reduce from 16 to 8

# Or use CPU
device='cpu',  # Change from 0 (GPU) to 'cpu'
```

---

#### 4. "API key not found"

**Cause**: .env file missing or incorrect

**Solution**:
```bash
cd backend
cat > .env << EOF
OPEN_AI_API_KEY=your_key_here
GROQ_AI_API_KEY=your_key_here
EOF
```

---

#### 5. "Too many detections / False positives"

**Cause**: Confidence threshold too low

**Solution**:
```python
# Edit backend/api_utils.py line 72
"Impacted Tooth": 0.40,  # Increase from 0.25

# Or adjust spatial filtering (line 145)
if not (center_x < 0.20 or center_x > 0.80):  # Stricter
```

---

#### 6. "Missing obvious impacted tooth"

**Cause**: Confidence threshold too high or spatial filtering too strict

**Solution**:
```python
# Edit backend/api_utils.py line 72
"Impacted Tooth": 0.20,  # Decrease from 0.25

# Relax spatial filtering (line 145)
if not (center_x < 0.35 or center_x > 0.65):  # More lenient
```

---

#### 7. "Gradio interface won't load"

**Cause**: Port already in use

**Solution**:
```bash
# Check what's using port 7860
lsof -i :7860

# Kill the process
kill <PID>

# Or use a different port
# Edit dental_ai_unified.py line ~790
demo.launch(server_port=7861)  # Change from 7860
```

---

### Debug Mode

Enable verbose logging:

```python
# Edit dental_ai_unified.py line ~10
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export DEBUG=True
python dental_ai_unified.py
```

---

## Development

### Adding a New AI Model

**1. Add API client** (`api_utils.py`):
```python
from new_api import NewAPIClient

def init_clients():
    ...
    new_client = NewAPIClient(api_key=os.getenv("NEW_API_KEY"))
    return openai_client, groq_client, new_client
```

**2. Add handler** (`api_utils.py`):
```python
async def chat_with_context_async(...):
    ...
    elif model_name == "new_model":
        response = await loop.run_in_executor(
            None,
            lambda: new_client.generate(messages)
        )
        return {"model": "new_model", "response": response.text, "success": True}
```

**3. Update routing** (`multimodal_utils.py`):
```python
mode, models = "vision", ["gpt4", "groq", "qwen", "new_model"]
```

**4. Update UI formatting** (`multimodal_utils.py`):
```python
new_model_text = responses.get('new_model', '')
# Add display box in format_multi_model_response()
```

---

### Modifying YOLO Classes

**1. Retrain model** with new dataset containing desired classes

**2. Update class thresholds** (`api_utils.py`):
```python
CLASS_CONFIDENCE_THRESHOLDS = {
    "New Class": 0.30,
    ...
}
```

**3. Update spatial filtering** if class has location constraints

---

### Custom Dataset Format

YOLO expects this structure:
```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt  # One box per line: class_id x_center y_center width height
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

**data.yaml**:
```yaml
path: /absolute/path/to/dataset
train: train/images
val: valid/images
test: test/images

names:
  0: Class1
  1: Class2
  2: Class3

nc: 3
```

---

### Running Tests

```bash
# Test YOLO detection
python -c "
from PIL import Image
from api_utils import detect_teeth_yolo

img = Image.open('sample_images/sample1.jpg')
result = detect_teeth_yolo(img)
print(result)
"

# Test API connections
python -c "
from api_utils import init_clients
import asyncio

async def test():
    openai_client, groq_client = init_clients()
    messages = [{'role': 'user', 'content': 'Hello'}]

    from api_utils import chat_with_context_async
    result = await chat_with_context_async(messages, 'gpt4', openai_client, groq_client)
    print(result)

asyncio.run(test())
"
```

---

## Credits & License

**Author**: Harsha
**Version**: 2.3
**Last Updated**: December 22, 2025

### Technologies Used

- **YOLOv8**: Ultralytics (AGPL-3.0)
- **PyTorch**: Meta AI (BSD)
- **Gradio**: Hugging Face (Apache 2.0)
- **OpenAI API**: OpenAI
- **Groq**: Groq Inc.

### Dataset

- **Roboflow**: dental-x-ray-1imfs by gozdes-projects (CC BY 4.0)

### Disclaimer

**This software is for educational and research purposes only.**

NOT intended for clinical diagnosis or treatment decisions. Always consult qualified dental professionals for medical advice.

---

## Quick Links

- **GitHub**: (Add your repository URL)
- **Issues**: (Add issues URL)
- **API Docs**: See [API Reference](#api-reference)
- **Training Guide**: See `docs/TRAINING_GUIDE.md`
- **Quick Reference**: See `docs/QUICK_REFERENCE.md`

---

**End of Documentation**

For questions or support, refer to the troubleshooting section or consult the codebase comments.
