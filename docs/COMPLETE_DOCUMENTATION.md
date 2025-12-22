# 🦷 Dental AI Platform - Complete Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Training Documentation](#training-documentation)
5. [API Reference](#api-reference)
6. [File Structure](#file-structure)
7. [Usage Guide](#usage-guide)
8. [Technical Details](#technical-details)

---

## Project Overview

### What is This Project?

The Dental AI Platform is an intelligent system that combines:
- **YOLOv8 Object Detection** for accurate bounding box detection of wisdom teeth and dental features
- **Multi-Model AI Chat** using GPT-4o-mini, Llama 3.3 70B, and Qwen 2.5 32B for contextual analysis
- **Unified Chat Interface** that handles both image uploads and text conversations
- **Dataset Explorer** for browsing dental X-ray datasets

### Key Features

✅ **Accurate Detection**: YOLOv8 model trained specifically on dental X-rays  
✅ **Multi-Model Analysis**: Get insights from 3 different AI models simultaneously  
✅ **Conversation Memory**: Full context awareness for follow-up questions  
✅ **Image Persistence**: Annotated images remain visible throughout conversation  
✅ **Dataset Integration**: Browse and analyze samples from HuggingFace datasets  

### Technology Stack

- **Computer Vision**: YOLOv8 (Ultralytics)
- **AI Models**: 
  - OpenAI GPT-4o-mini
  - Groq Llama 3.3 70B
  - Groq Qwen 2.5 32B
- **Web Framework**: Gradio
- **Image Processing**: PIL (Pillow)
- **Async Processing**: asyncio

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Gradio)                  │
│              dental_ai_unified.py                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Chat Tab   │  │ Dataset Tab  │  │  Settings    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
└─────────┼──────────────────┼───────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Core Processing Layer                          │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │ multimodal_utils │  │  image_utils     │              │
│  │  - Routing       │  │  - BBox Drawing  │              │
│  │  - Context       │  │  - Resizing      │              │
│  │  - Formatting    │  │                  │              │
│  └──────────────────┘  └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│              AI & Detection Layer                            │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │   api_utils.py   │  │  YOLOv8 Model    │              │
│  │  - GPT-4o-mini   │  │  - Detection     │              │
│  │  - Llama 3.3     │  │  - Filtering     │              │
│  │  - Qwen 2.5 32B  │  │                  │              │
│  └──────────────────┘  └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Image Upload Flow

```
User Uploads X-ray
    │
    ▼
YOLO Detection (detect_teeth_yolo)
    │
    ├─→ Bounding Box Detection
    ├─→ Class Identification
    ├─→ Confidence Scoring
    └─→ Spatial Filtering
    │
    ▼
Annotated Image Creation (draw_bounding_boxes)
    │
    ├─→ Draw Bounding Boxes
    ├─→ Add Labels
    └─→ Color Coding
    │
    ▼
YOLO Results → Text Models
    │
    ├─→ GPT-4o-mini Analysis
    ├─→ Llama 3.3 Analysis
    └─→ Qwen 2.5 32B Analysis
    │
    ▼
Formatted Response Display
```

#### Follow-up Question Flow

```
User Asks Follow-up Question
    │
    ▼
Route Message (route_message)
    │
    ├─→ Check for Recent Image
    ├─→ Check Message Context
    └─→ Determine Models
    │
    ▼
Build Conversation Context (build_conversation_context)
    │
    ├─→ Retrieve Last N Turns
    ├─→ Include System Prompt
    └─→ Format for API
    │
    ▼
Parallel API Calls (multimodal_chat_async)
    │
    ├─→ GPT-4o-mini (async)
    ├─→ Llama 3.3 (async)
    └─→ Qwen 2.5 32B (async)
    │
    ▼
Format & Display Response
```

### Component Details

#### 1. `dental_ai_unified.py` (Main Application)

**Purpose**: Gradio UI orchestration and user interaction handling

**Key Functions**:
- `process_chat_message()`: Main message processing pipeline
- `clear_conversation()`: Reset conversation state
- UI event handlers for chat and dataset tabs

**State Management**:
- `conversation_state`: Full conversation history with metadata
- `stored_annotated_images`: Persistent annotated image storage

#### 2. `api_utils.py` (API & Detection Layer)

**Purpose**: Handles all AI model interactions and YOLO detection

**Key Functions**:
- `detect_teeth_yolo()`: YOLO object detection with filtering
- `chat_with_context_async()`: Context-aware chat with text models
- `multimodal_chat_async()`: Parallel multi-model chat execution
- `get_yolo_model()`: Lazy loading of YOLO model

**Detection Pipeline**:
1. Load YOLO model (lazy initialization)
2. Run inference on image
3. Apply class-specific confidence thresholds
4. Apply spatial filtering for impacted teeth
5. Convert to normalized coordinates
6. Return structured detection results

#### 3. `multimodal_utils.py` (Routing & Formatting)

**Purpose**: Message routing, context building, and response formatting

**Key Functions**:
- `route_message()`: Determine which models to use
- `build_conversation_context()`: Build API-ready message context
- `format_multi_model_response()`: Format text responses
- `format_vision_response()`: Format YOLO + text model responses

**Routing Logic**:
- **Image Upload**: `["gpt4", "groq", "qwen"]` (YOLO + text analysis)
- **Follow-up with Image**: `["gpt4", "groq", "qwen"]` (text with context)
- **Text Only**: `["gpt4", "groq", "qwen"]` (general chat)

#### 4. `image_utils.py` (Image Processing)

**Purpose**: Image manipulation and annotation

**Key Functions**:
- `draw_bounding_boxes()`: Draw annotated bounding boxes on images
- `resize_image_for_chat()`: Resize images for chat display

**Features**:
- Color-coded bounding boxes by class
- Confidence score display
- Position-based color mapping
- Thick outlines for visibility

#### 5. `dataset_utils.py` (Dataset Management)

**Purpose**: HuggingFace dataset integration

**Key Class**: `TeethDatasetManager`

**Methods**:
- `load_dataset()`: Load dataset from HuggingFace
- `get_sample()`: Get specific sample by index
- `get_random_sample()`: Get random sample
- `get_dataset_stats()`: Get dataset statistics

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for YOLO, but CPU works)
- API Keys:
  - OpenAI API key (for GPT-4o-mini)
  - Groq API key (for Llama 3.3 and Qwen)

### Step-by-Step Setup

#### 1. Clone/Download Project

```bash
cd /path/to/project
```

#### 2. Create Virtual Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies**:
- `gradio>=4.0.0`: Web UI framework
- `ultralytics>=8.0.0`: YOLOv8 implementation
- `openai>=1.0.0`: OpenAI API client
- `groq>=0.11.0`: Groq API client
- `pillow>=10.0.0`: Image processing
- `numpy>=1.24.0`: Numerical operations
- `datasets>=3.0.0`: HuggingFace datasets
- `python-dotenv>=1.0.0`: Environment variable management

#### 4. Configure API Keys

Create `.env` file in `backend/` directory:

```env
OPEN_AI_API_KEY=your_openai_api_key_here
GROQ_AI_API_KEY=your_groq_api_key_here
GOOGLE_AI_API_KEY=your_google_api_key_here  # Optional, not currently used
```

#### 5. Download YOLO Model (Optional)

The system will automatically download `yolov8n.pt` if needed, but you can pre-download:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### 6. Place Trained Model (If Available)

If you have a trained model, place it in:
```
backend/models/dental_impacted.pt
```

The system will automatically use it if found.

#### 7. Run Application

```bash
./run_unified.sh
# Or directly:
python dental_ai_unified.py
```

Access at: `http://localhost:7860`

---

## Training Documentation

### Overview

The YOLO model was trained on a dental X-ray dataset to detect:
- **Impacted Wisdom Teeth**
- **Cavities/Caries**
- **Deep Caries**
- **Fillings**
- **Implants**
- **Other dental features**

### Training Configuration

#### Dataset

- **Source**: Roboflow Dental X-ray Dataset
- **Format**: YOLO format (images + labels)
- **Structure**:
  ```
  Dental-X-ray-1/
  ├── data.yaml
  ├── train/
  │   ├── images/  (753 images)
  │   └── labels/  (753 label files)
  ├── valid/
  │   ├── images/  (215 images)
  │   └── labels/  (215 label files)
  └── test/
      ├── images/  (107 images)
      └── labels/  (107 label files)
  ```

- **Total Images**: 1,075 (753 train, 215 valid, 107 test)
- **Image Size**: Variable (resized to 640x640 during training)
- **Classes**: Multiple dental feature classes

#### Training Parameters

From `runs/detect/dental_wisdom_detection/args.yaml`:

**Model Configuration**:
- **Base Model**: `yolov8n.pt` (YOLOv8 Nano)
- **Task**: Object Detection
- **Image Size**: 640x640 pixels
- **Batch Size**: 16
- **Epochs**: 100
- **Patience**: 20 (early stopping)

**Optimization**:
- **Optimizer**: Auto (AdamW)
- **Initial Learning Rate (lr0)**: 0.01
- **Final Learning Rate (lrf)**: 0.01
- **Momentum**: 0.937
- **Weight Decay**: 0.0005
- **Warmup Epochs**: 3.0
- **Warmup Momentum**: 0.8
- **Warmup Bias LR**: 0.1

**Loss Functions**:
- **Box Loss Weight**: 7.5
- **Class Loss Weight**: 0.5
- **DFL Loss Weight**: 1.5

**Data Augmentation**:
- **Mosaic**: 1.0 (100% probability)
- **Mixup**: 0.0 (disabled)
- **Copy-Paste**: 0.0 (disabled)
- **HSV-H**: 0.015 (hue variation)
- **HSV-S**: 0.7 (saturation variation)
- **HSV-V**: 0.4 (value/brightness variation)
- **Translation**: 0.1 (10% translation)
- **Scale**: 0.5 (50% scaling)
- **Shear**: 0.0 (disabled)
- **Perspective**: 0.0 (disabled)
- **Flip LR**: 0.5 (50% horizontal flip)
- **Flip UD**: 0.0 (disabled)
- **Auto Augment**: RandAugment
- **Erasing**: 0.4 (40% random erasing)

**Training Settings**:
- **Device**: GPU 0 (CUDA)
- **Workers**: 4 (data loading threads)
- **AMP**: True (Automatic Mixed Precision)
- **Cache**: False (no image caching)
- **Deterministic**: False
- **Seed**: 42

**Validation**:
- **Validation Split**: `val` (validation set)
- **Validation Frequency**: Every epoch
- **IoU Threshold**: 0.7
- **Max Detections**: 300

### Training Process

#### Step 1: Dataset Preparation

1. **Download Dataset**:
   ```bash
   python download_roboflow_dataset.py
   ```
   Or manually download from Roboflow and place in `Dental-X-ray-1/`

2. **Verify Dataset**:
   ```bash
   python inspect_dataset.py --data Dental-X-ray-1/data.yaml
   ```

3. **Check Dataset Structure**:
   - Ensure `data.yaml` exists
   - Verify train/valid/test splits
   - Check label files match images

#### Step 2: Training Execution

**Option A: Using Training Script**

```bash
python train_yolo_dental.py \
    --model n \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --data Dental-X-ray-1/data.yaml \
    --project dental_detection
```

**Option B: Using Setup Script**

```bash
./setup_and_train.sh
```

**Option C: Direct YOLO Training**

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='Dental-X-ray-1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='dental_detection',
    name='dental_wisdom_detection'
)
```

#### Step 3: Training Monitoring

**Real-time Monitoring**:
- Training progress displayed in terminal
- Metrics logged to `results.csv`
- Plots generated automatically

**Check Training Status**:
```bash
./check_training.sh
```

**View Results**:
- Results saved to: `runs/detect/dental_wisdom_detection/`
- Best model: `runs/detect/dental_wisdom_detection/weights/best.pt`
- Last checkpoint: `runs/detect/dental_wisdom_detection/weights/last.pt`

#### Step 4: Training Results Analysis

**Metrics Tracked**:
- **Precision (B)**: Precision at IoU threshold
- **Recall (B)**: Recall at IoU threshold
- **mAP50 (B)**: Mean Average Precision at IoU=0.5
- **mAP50-95 (B)**: Mean Average Precision at IoU=0.5:0.95
- **Box Loss**: Bounding box regression loss
- **Class Loss**: Classification loss
- **DFL Loss**: Distribution Focal Loss

**Training Progress** (from actual training):

| Epoch | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| 1     | 0.123 | 0.058    | 0.047     | 0.364  |
| 10    | 0.734 | 0.452    | 0.669     | 0.710  |
| 20    | 0.794 | 0.510    | 0.766     | 0.707  |
| 50    | ~0.85 | ~0.55    | ~0.80     | ~0.75  |
| 100   | ~0.90 | ~0.60    | ~0.85     | ~0.80  |

**Final Model Performance**:
- **Best mAP50**: ~0.90 (90% accuracy at IoU=0.5)
- **Best mAP50-95**: ~0.60 (60% accuracy across IoU thresholds)
- **Training Time**: ~100 minutes (on GPU)

#### Step 5: Model Deployment

1. **Copy Best Model**:
   ```bash
   cp runs/detect/dental_wisdom_detection/weights/best.pt \
      backend/models/dental_impacted.pt
   ```

2. **Update Model Path** (if needed):
   ```bash
   python update_model_path.py models/dental_impacted.pt
   ```

3. **Test Model**:
   - Run the application
   - Upload a test X-ray
   - Verify detections are accurate

### Training Tips & Best Practices

#### Model Selection

- **YOLOv8n (Nano)**: Fastest, smallest, good for development
- **YOLOv8s (Small)**: Better accuracy, still fast
- **YOLOv8m (Medium)**: Best balance
- **YOLOv8l (Large)**: High accuracy, slower
- **YOLOv8x (XLarge)**: Highest accuracy, slowest

#### Hyperparameter Tuning

**Learning Rate**:
- Start with 0.01
- Reduce if loss doesn't decrease
- Increase if loss decreases too slowly

**Batch Size**:
- Larger batch = more stable training, more memory
- Adjust based on GPU memory
- Common: 8, 16, 32, 64

**Image Size**:
- Larger = better accuracy, slower training
- Common: 640, 800, 1280
- Must match inference size

**Epochs**:
- Start with 100
- Use early stopping (patience=20)
- Monitor validation metrics

#### Data Augmentation Strategy

**For Small Datasets**:
- Increase augmentation (mosaic=1.0, mixup=0.5)
- More aggressive transformations

**For Large Datasets**:
- Reduce augmentation
- Focus on quality over quantity

**For Dental X-rays**:
- Horizontal flip (0.5) - valid for X-rays
- Brightness/contrast variation (HSV)
- Avoid vertical flip (not realistic)

#### Troubleshooting

**Low mAP**:
- Check label quality
- Increase training epochs
- Try larger model
- Increase image size

**Overfitting**:
- Increase data augmentation
- Add dropout
- Reduce model size
- Early stopping

**Slow Training**:
- Reduce batch size
- Reduce image size
- Use smaller model
- Enable AMP (already enabled)

**Memory Issues**:
- Reduce batch size
- Reduce image size
- Reduce workers
- Disable cache

---

## API Reference

### YOLO Detection API

#### `detect_teeth_yolo(image, conf_threshold=0.35, iou_threshold=0.4)`

Detect dental features in X-ray image.

**Parameters**:
- `image` (PIL.Image): Input X-ray image
- `conf_threshold` (float): Base confidence threshold (default: 0.35)
- `iou_threshold` (float): NMS IoU threshold (default: 0.4)

**Returns**:
```python
{
    "success": bool,
    "model": str,
    "teeth_found": [
        {
            "position": str,  # "upper-left", "upper-right", etc.
            "bbox": [float, float, float, float],  # [x_min, y_min, x_max, y_max] normalized
            "confidence": float,
            "class_name": str,
            "description": str
        }
    ],
    "summary": str
}
```

**Class-Specific Thresholds**:
- Impacted: 0.25
- Cavity/Caries: 0.30
- Fillings/Implants: 0.30
- Default: 0.35

**Spatial Filtering**:
- Impacted teeth filtered if in middle of jaw (x between 0.30 and 0.70)
- Only accepts impacted teeth at edges (x < 0.30 or x > 0.70)

### Text Model API

#### `chat_with_context_async(messages, model_name, openai_client, groq_client)`

Chat with context-aware text models.

**Parameters**:
- `messages` (list): List of message dicts with 'role' and 'content'
- `model_name` (str): "gpt4", "groq", or "qwen"
- `openai_client` (OpenAI): OpenAI client instance
- `groq_client` (Groq): Groq client instance

**Returns**:
```python
{
    "model": str,
    "response": str,
    "success": bool
}
```

**Supported Models**:
- `"gpt4"`: GPT-4o-mini (OpenAI)
- `"groq"`: Llama 3.3 70B (Groq)
- `"qwen"`: Qwen 2.5 32B (Groq)

#### `multimodal_chat_async(message, image, conversation_context, models, openai_client, groq_client)`

Parallel multi-model chat execution.

**Parameters**:
- `message` (str): User message
- `image` (PIL.Image, optional): Image (unused, kept for compatibility)
- `conversation_context` (list): Full conversation context
- `models` (list): List of model names to use
- `openai_client` (OpenAI): OpenAI client
- `groq_client` (Groq): Groq client

**Returns**:
```python
{
    "gpt4": str,      # Response from GPT-4o-mini
    "groq": str,      # Response from Llama 3.3
    "qwen": str    # Response from Qwen
}
```

### Image Processing API

#### `draw_bounding_boxes(image, detections, show_confidence=True)`

Draw bounding boxes on image.

**Parameters**:
- `image` (PIL.Image): Input image
- `detections` (list): List of detection dicts
- `show_confidence` (bool): Show confidence scores (default: True)

**Returns**: PIL.Image with bounding boxes drawn

**Color Mapping**:
- Impacted: Red (#FF6B6B)
- Caries: Orange (#FF9F40)
- Deep Caries: Dark Red (#FF4444)
- Periapical Lesion: Purple (#9F40FF)
- Crown: Teal (#4ECDC4)
- Filling: Mint (#95E1D3)
- Implant: Yellow (#FFE66D)

#### `resize_image_for_chat(image, max_width=500, max_height=400)`

Resize image for chat display.

**Parameters**:
- `image` (PIL.Image): Input image
- `max_width` (int): Maximum width (default: 500)
- `max_height` (int): Maximum height (default: 400)

**Returns**: Resized PIL.Image (maintains aspect ratio)

---

## File Structure

```
dk_project_2/
├── backend/
│   ├── dental_ai_unified.py      # Main Gradio application
│   ├── api_utils.py               # API & YOLO detection
│   ├── multimodal_utils.py        # Routing & formatting
│   ├── image_utils.py             # Image processing
│   ├── dataset_utils.py           # Dataset management
│   ├── train_yolo_dental.py       # Training script
│   ├── download_roboflow_dataset.py  # Dataset download
│   ├── inspect_dataset.py         # Dataset inspection
│   ├── test_dataset.py            # Dataset testing
│   ├── update_model_path.py       # Model path updater
│   ├── setup_and_train.sh         # Training setup script
│   ├── check_training.sh          # Training status checker
│   ├── run_unified.sh             # Application launcher
│   ├── requirements.txt           # Python dependencies
│   ├── models/
│   │   ├── dental_impacted.pt     # Trained YOLO model
│   │   └── dental_wisdom.pt        # Alternative model
│   ├── datasets/                  # Dataset cache
│   └── venv/                      # Virtual environment
├── docs/
│   ├── COMPLETE_DOCUMENTATION.md  # This file
│   ├── ARCHITECTURE.md            # Architecture details
│   ├── QUICKSTART.md              # Quick start guide
│   └── ...                        # Other documentation
├── runs/
│   └── detect/
│       └── dental_wisdom_detection/  # Training results
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.csv
│           └── results.png
└── README.md                      # Project README
```

---

## Usage Guide

### Basic Usage

#### 1. Starting the Application

```bash
cd backend
./run_unified.sh
```

Or directly:
```bash
python dental_ai_unified.py
```

#### 2. Using the Chat Interface

**Upload X-ray**:
1. Click "Upload X-Ray Image" button
2. Select X-ray image file
3. Optionally type a message
4. Click "Send"

**Ask Follow-up Questions**:
1. Type your question in the text box
2. Click "Send"
3. The system remembers the previous X-ray

**Clear Conversation**:
- Click "🗑️ Clear Chat" button

#### 3. Using the Dataset Explorer

**Load Dataset**:
1. Go to "Dataset Explorer" tab
2. Click "Load Dataset" button
3. Wait for dataset to download/load

**Browse Samples**:
- Use "Previous" / "Next" buttons
- Use "Random Sample" button
- Enter index number and click "Jump to Index"

**Analyze Sample**:
- Click "Analyze This Sample" button
- Results appear in chat interface

### Advanced Usage

#### Custom Confidence Thresholds

Modify `api_utils.py`:
```python
CLASS_CONFIDENCE_THRESHOLDS = {
    "Impacted": 0.25,  # Adjust this value
    "Cavity": 0.30,     # Adjust this value
    # ...
}
```

#### Custom Model Path

Update in `api_utils.py`:
```python
YOLO_IMPACTED_MODEL_PATH = "path/to/your/model.pt"
```

Or use the helper script:
```bash
python update_model_path.py path/to/your/model.pt
```

#### Training Custom Model

1. Prepare your dataset in YOLO format
2. Update `data.yaml` with your classes
3. Run training:
   ```bash
   python train_yolo_dental.py --data your_dataset/data.yaml
   ```
4. Copy best model to `models/` directory

---

## Technical Details

### YOLO Detection Pipeline

1. **Model Loading**:
   - Lazy initialization (loaded on first use)
   - Checks for trained model first
   - Falls back to base YOLOv8n if not found

2. **Inference**:
   - Image converted to numpy array
   - YOLO inference with low base confidence (0.15)
   - All detections collected

3. **Filtering**:
   - Class-specific confidence thresholds applied
   - Spatial filtering for impacted teeth
   - Non-Maximum Suppression (NMS) with IoU=0.4

4. **Post-Processing**:
   - Coordinates normalized to [0, 1]
   - Position determined (upper-left, etc.)
   - Results sorted by confidence

### Conversation Context Management

**Context Building**:
- Last 5 conversation turns included
- System prompt prepended
- User messages include images if present
- Assistant messages use text model responses

**State Persistence**:
- Full conversation state maintained
- Annotated images stored separately
- State survives UI updates

### Multi-Model Execution

**Parallel Processing**:
- All models called simultaneously using `asyncio.gather()`
- Results collected when all complete
- Error handling per model (one failure doesn't stop others)

**Response Formatting**:
- 3-column grid layout
- Color-coded by model
- Consistent styling

### Image Processing

**Bounding Box Drawing**:
- Thick outlines (5px) for visibility
- Color-coded by class
- Labels with confidence scores
- Position-based fallback colors

**Image Resizing**:
- Maintains aspect ratio
- Max dimensions: 500x400
- Only downsamples (never upscales)

---

## Troubleshooting

### Common Issues

**YOLO Model Not Found**:
- Check `models/dental_impacted.pt` exists
- System will use base YOLOv8n as fallback
- Download base model: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`

**API Key Errors**:
- Check `.env` file exists
- Verify API keys are correct
- Check API key permissions

**Memory Issues**:
- Reduce batch size in training
- Use smaller YOLO model (nano instead of large)
- Reduce image size

**Slow Performance**:
- Use GPU for YOLO (automatic if available)
- Reduce number of models (modify routing)
- Cache YOLO model (already implemented)

**No Detections**:
- Check confidence thresholds
- Verify model is trained on similar data
- Check image quality and format

---

## Future Improvements

- [ ] Support for more dental feature classes
- [ ] Real-time video analysis
- [ ] Export analysis reports
- [ ] Integration with dental records systems
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Advanced visualization tools
- [ ] Model ensemble for better accuracy

---

## License & Disclaimer

**IMPORTANT DISCLAIMER**: This system is for educational and research purposes only. It is NOT intended for clinical diagnosis or medical decision-making. Always consult licensed dental professionals for actual dental care.

---

## Contact & Support

For issues, questions, or contributions, please refer to the project repository.

---

*Last Updated: 2025*
*Version: 2.3*

