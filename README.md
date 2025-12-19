# 🦷 Dental AI Platform

An advanced AI-powered dental analysis platform featuring wisdom tooth detection and multi-model chatbot capabilities, built with Gradio.

## 🌟 Features

### 🔍 Tab 1: Wisdom Tooth Detection
- Upload dental X-ray images for AI analysis
- Detect wisdom teeth using GPT-4o Vision or Gemini Vision
- Automatic bounding box annotation with color-coded positions
- Detailed analysis with tooth descriptions and coordinates
- Side-by-side comparison of original and annotated images

### 💬 Tab 2: Multi-Model Chatbot
- Query 3 AI models simultaneously in parallel:
  - **OpenAI GPT-4o** - Advanced reasoning
  - **Google Gemini 1.5 Flash** - Fast responses
  - **Groq Llama3 70B** - Ultra-fast inference
- Compare responses side by side
- Example questions included
- Async execution for optimal performance

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
cd backend
./setup.sh
source venv/bin/activate
python dental_ai_app.py
```

### Option 2: Manual Installation
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python dental_ai_app.py
```

### Option 3: Quick Run (if already set up)
```bash
cd backend
./run.sh
```

Then open your browser to: **http://localhost:7860**

## 📁 Project Structure

```
dk_project_2/
├── backend/
│   ├── dental_ai_app.py      # Main Gradio application with 2 tabs
│   ├── api_utils.py          # API functions (vision + chat models)
│   ├── image_utils.py        # Image processing & annotation
│   ├── requirements.txt      # Python dependencies
│   ├── .env                  # API keys (DO NOT COMMIT!)
│   ├── setup.sh              # Automated setup script
│   ├── run.sh                # Quick run script
│   └── DENTAL_AI_README.md   # Detailed documentation
├── .gitignore
└── README.md                 # This file
```

## 🔑 API Keys Setup

Create/update `backend/.env` with your API keys:

```env
OPEN_AI_API_KEY=sk-proj-...
GROQ_AI_API_KEY=gsk_...
GOOGLE_AI_API_KEY=AIza...
```

### Get Your API Keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Groq**: https://console.groq.com/keys (Free tier available!)
- **Google AI**: https://makersuite.google.com/app/apikey

## 💻 Tech Stack

- **Gradio** - Web UI framework
- **OpenAI API** - GPT-4o & GPT-4o Vision
- **Groq API** - Ultra-fast Llama3 inference
- **Google Generative AI** - Gemini & Gemini Vision
- **PIL/OpenCV** - Image processing
- **AsyncIO** - Concurrent API calls

## 📖 Usage

### Wisdom Tooth Detection
1. Upload a dental X-ray image (panoramic works best)
2. Select AI model (GPT-4o Vision or Gemini Vision)
3. Click "Analyze X-Ray"
4. View annotated image with bounding boxes
5. Read detailed analysis

### Multi-Model Chatbot
1. Type your question or select an example
2. Click "Ask All Models" or press Enter
3. Wait 3-8 seconds for parallel responses
4. Compare answers from all 3 models side by side

## 📚 Documentation

For detailed documentation, see:
- **[backend/DENTAL_AI_README.md](backend/DENTAL_AI_README.md)** - Complete guide with:
  - Architecture details
  - Customization guide
  - Troubleshooting
  - Performance optimization
  - API comparison
  - Security notes

## 🎯 Key Highlights

✅ **Modular Architecture** - Separated concerns (API, image processing, UI)
✅ **Async Execution** - Parallel API calls for 3x faster responses
✅ **Error Handling** - Graceful failures with helpful error messages
✅ **Beautiful UI** - Custom CSS with gradient headers and responsive layout
✅ **Production Ready** - Virtual environment, gitignore, comprehensive docs

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` |
| Invalid API key | Check `.env` file, verify keys on provider sites |
| Port already in use | Change `server_port` in `dental_ai_app.py` |
| Slow responses | Normal for vision APIs (10-15s), use Groq for speed |

## 🔒 Security

- ✅ `.env` file in `.gitignore` (never commit API keys!)
- ✅ Environment variables for sensitive data
- ✅ Error messages don't expose keys
- ⚠️ Add rate limiting for public deployments

## 📊 Model Comparison

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| GPT-4o Vision | Slow (10-15s) | Excellent | $$$ | Critical analysis |
| Gemini Vision | Fast (5-8s) | Very Good | $ | Quick checks |
| GPT-4o Chat | Medium | Excellent | $$ | Complex reasoning |
| Gemini Chat | Fast | Very Good | $ | General queries |
| Groq Llama3 | Very Fast (2-3s) | Good | Free | Speed priority |

## 🤝 Contributing

Contributions welcome! Areas to extend:
- Add more vision models (Claude Vision, LLaVA)
- Implement response caching
- Add export functionality (PDF reports)
- Create batch processing for multiple X-rays
- Add treatment recommendation system

## 📄 License

MIT License - Free to use and modify

---

**Built with ❤️ for dental AI applications**
