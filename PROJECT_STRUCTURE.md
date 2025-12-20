# 📁 Project Structure

Complete overview of the Dental AI Platform file organization.

```
dk_project_2/
│
├── 📂 backend/                          # Application code
│   ├── 🐍 dental_ai_app.py             # Basic app (2 tabs)
│   ├── 🐍 dental_ai_enhanced.py        # Enhanced app (3 tabs + dataset) ⭐ RECOMMENDED
│   ├── 🐍 api_utils.py                 # API integration layer
│   ├── 🐍 image_utils.py               # Image processing utilities
│   ├── 🐍 dataset_utils.py             # HuggingFace dataset manager ⭐
│   ├── 🐍 app.py                       # Legacy chatbot (deprecated)
│   ├── 🐍 test_example.py              # API connectivity tests
│   │
│   ├── 📄 requirements.txt             # Python dependencies
│   ├── 🔐 .env                         # API keys (DO NOT COMMIT!)
│   │
│   ├── 🔧 setup.sh                     # Automated installation script
│   ├── 🚀 run.sh                       # Run basic app
│   ├── 🚀 run_enhanced.sh              # Run enhanced app ⭐
│   │
│   └── 📖 README.md                    # Backend-specific docs
│
├── 📂 docs/                             # Documentation hub ⭐
│   ├── 📘 README.md                    # Documentation index & navigation
│   ├── 📘 QUICKSTART.md                # Quick start guide (30s setup)
│   ├── 📘 DENTAL_AI_README.md          # Complete user guide (Tabs 1 & 2)
│   ├── 📘 DATASET_FEATURES.md          # Dataset integration guide (Tab 3)
│   └── 📘 ARCHITECTURE.md              # Technical architecture & design
│
├── 📂 frontend/                         # (Empty - Gradio handles UI)
│
├── 📄 README.md                         # Main project README
├── 📄 PROJECT_STRUCTURE.md              # This file
├── 📄 .gitignore                        # Git ignore rules
└── 📄 ai_usage_log.md                   # Usage tracking


Generated files (not in git):
├── backend/venv/                        # Virtual environment
└── ~/.cache/huggingface/                # Dataset cache
```

---

## 📊 File Statistics

### Application Code
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| dental_ai_enhanced.py | ~380 | Main app with 3 tabs | ✅ Recommended |
| dental_ai_app.py | ~300 | Basic app with 2 tabs | ✅ Active |
| api_utils.py | ~240 | API integrations | ✅ Active |
| image_utils.py | ~180 | Image processing | ✅ Active |
| dataset_utils.py | ~320 | Dataset management | ✅ Active |
| app.py | ~180 | Legacy chatbot | ⚠️ Deprecated |
| test_example.py | ~100 | Testing utilities | ✅ Active |

**Total Application Code:** ~1,700 lines

### Documentation
| File | Lines | Topics |
|------|-------|--------|
| DENTAL_AI_README.md | ~400 | User guide, setup, usage |
| DATASET_FEATURES.md | ~500 | Dataset integration |
| ARCHITECTURE.md | ~750 | System design, architecture |
| QUICKSTART.md | ~200 | Quick reference |
| docs/README.md | ~150 | Navigation index |

**Total Documentation:** ~2,000 lines

---

## 🎯 File Purposes

### Core Application Files

#### `dental_ai_enhanced.py` ⭐ **RECOMMENDED**
```python
# Enhanced version with 3 tabs
- Tab 1: Wisdom Tooth Detection (Vision AI)
- Tab 2: Multi-Model Chatbot (3 models parallel)
- Tab 3: Dataset Explorer (1,206 samples)
```

**Use when:** You want full features including dataset browsing

#### `dental_ai_app.py`
```python
# Basic version with 2 tabs
- Tab 1: Wisdom Tooth Detection
- Tab 2: Multi-Model Chatbot
```

**Use when:** You only need core AI features without dataset

#### `api_utils.py`
```python
# API integration layer
- Vision APIs: GPT-4o Vision, Gemini Vision
- Chat APIs: OpenAI, Gemini, Groq (async)
- Error handling and response formatting
```

**Imported by:** dental_ai_app.py, dental_ai_enhanced.py, test_example.py

#### `image_utils.py`
```python
# Image processing utilities
- JSON parsing from AI responses
- Bounding box drawing with PIL
- Image annotation and labeling
- Color-coded tooth positions
```

**Imported by:** dental_ai_app.py, dental_ai_enhanced.py

#### `dataset_utils.py` ⭐
```python
# HuggingFace dataset integration
Class: TeethDatasetManager
- Load dataset from HuggingFace
- Browse samples (next/prev/random)
- Batch processing
- Export results
```

**Imported by:** dental_ai_enhanced.py

---

## 🚀 Scripts & Utilities

### Setup & Installation

#### `setup.sh`
```bash
# Automated setup script
1. Check Python version
2. Install pip if needed
3. Create virtual environment
4. Install dependencies
5. Display success message
```

**Usage:** `./setup.sh` (one-time setup)

#### `requirements.txt`
```
# Python dependencies
gradio==4.44.0                # Web UI
openai==1.54.0                # GPT-4o API
groq==0.11.0                  # Groq API
google-generativeai==0.8.3    # Gemini API
datasets==3.1.0               # HuggingFace ⭐
pillow, opencv-python, numpy, etc.
```

### Launch Scripts

#### `run_enhanced.sh` ⭐
```bash
# Launch enhanced app (3 tabs)
python dental_ai_enhanced.py
```

**Usage:** `./run_enhanced.sh`

#### `run.sh`
```bash
# Launch basic app (2 tabs)
python dental_ai_app.py
```

**Usage:** `./run.sh`

### Testing

#### `test_example.py`
```python
# Test suite for API connectivity
- Check API keys loaded
- Test client initialization
- Run simple chat tests
```

**Usage:** `python test_example.py`

---

## 📚 Documentation Files

### `/docs/README.md`
**Navigation hub** for all documentation with:
- Quick links by topic
- Documentation for different user types
- Topic index

### `/docs/QUICKSTART.md`
**Quick reference** including:
- 30-second setup
- File overview
- Architecture diagram
- Customization tips
- Cost estimates

### `/docs/DENTAL_AI_README.md`
**Complete user guide** covering:
- Tabs 1 & 2 detailed usage
- Installation & setup
- Model comparison
- Troubleshooting
- Advanced features

### `/docs/DATASET_FEATURES.md`
**Dataset integration guide** with:
- Tab 3 usage (Dataset Explorer)
- Batch processing
- Performance tips
- Code examples
- API usage tracking

### `/docs/ARCHITECTURE.md`
**Technical documentation** featuring:
- System architecture
- Component breakdown
- Data flow diagrams
- API integration details
- Performance optimization
- Extension points

---

## 🔐 Environment & Config Files

### `.env` (backend/)
```env
# API Keys - NEVER COMMIT!
OPEN_AI_API_KEY=sk-proj-...
GROQ_AI_API_KEY=gsk_...
GOOGLE_AI_API_KEY=AIza...
```

**Location:** `backend/.env`
**Status:** In `.gitignore` ✅

### `.gitignore`
```gitignore
# Prevents committing:
*.env
__pycache__/
venv/
*.pyc
.DS_Store
```

---

## 📦 Generated/Cache Files (Not in Git)

### `backend/venv/`
- Python virtual environment
- Created by: `setup.sh` or `python -m venv venv`
- Size: ~100-200 MB

### `~/.cache/huggingface/datasets/`
- HuggingFace dataset cache
- Contains: RayanAi/Main_teeth_dataset (90 MB)
- Auto-downloaded on first dataset load

---

## 🗺️ User Workflows by File

### First-Time Setup
```
1. cd backend
2. ./setup.sh              # Creates venv, installs deps
3. source venv/bin/activate
4. Edit .env               # Add API keys
5. python test_example.py  # Verify setup
```

### Daily Usage - Enhanced App
```
1. cd backend
2. ./run_enhanced.sh       # Launches app
3. Open http://localhost:7860
```

### Daily Usage - Basic App
```
1. cd backend
2. ./run.sh                # Launches basic app
3. Open http://localhost:7860
```

### Development Workflow
```
1. Edit: dental_ai_enhanced.py, api_utils.py, etc.
2. Test: python test_example.py
3. Run: python dental_ai_enhanced.py
4. Debug: Check console output
```

---

## 🔄 File Dependencies

```
dental_ai_enhanced.py
    ├── api_utils.py
    │   ├── openai (external)
    │   ├── groq (external)
    │   └── google.generativeai (external)
    ├── image_utils.py
    │   ├── PIL (external)
    │   ├── cv2 (external)
    │   └── numpy (external)
    ├── dataset_utils.py
    │   ├── datasets (external - HuggingFace)
    │   ├── api_utils.py
    │   └── image_utils.py
    └── gradio (external)

dental_ai_app.py
    ├── api_utils.py
    ├── image_utils.py
    └── gradio (external)
```

---

## 📈 Growth & Maintenance

### Recently Added ⭐
- `dental_ai_enhanced.py` - Enhanced app with dataset
- `dataset_utils.py` - Dataset management
- `/docs/` directory - Organized documentation
- `DATASET_FEATURES.md` - Dataset guide
- `run_enhanced.sh` - Enhanced launcher

### Deprecated ⚠️
- `app.py` - Old chatbot (replaced by dental_ai_app.py)

### Future Additions (Planned)
- `/models/` - Custom trained models
- `/exports/` - Batch analysis results
- `/tests/` - Unit test suite
- `docker-compose.yml` - Docker deployment

---

## 🎨 Color Legend

- 🐍 Python source files
- 📄 Configuration/text files
- 🔧 Setup scripts
- 🚀 Run scripts
- 📖 Basic documentation
- 📘 Comprehensive documentation
- 📂 Directories
- ⭐ Recommended/Featured
- ⚠️ Deprecated/Warning
- ✅ Active/Working
- 🔐 Sensitive/Secret

---

## 📞 Quick Reference

| Need | File | Command |
|------|------|---------|
| Run enhanced app | dental_ai_enhanced.py | `./run_enhanced.sh` |
| Run basic app | dental_ai_app.py | `./run.sh` |
| Setup project | setup.sh | `./setup.sh` |
| Test APIs | test_example.py | `python test_example.py` |
| Read docs | docs/README.md | Open in editor |
| Quick start | docs/QUICKSTART.md | Reference guide |
| Architecture | docs/ARCHITECTURE.md | Technical details |

---

**Project Version:** 1.0 Enhanced
**Last Updated:** December 2024
**Total Files:** 20+ (excluding generated)
**Total Lines:** ~3,700 (code + docs)
