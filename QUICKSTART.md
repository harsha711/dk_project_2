# 🚀 Quick Start Guide - Dental AI Platform

## 30-Second Setup

```bash
cd backend
./setup.sh
source venv/bin/activate
python dental_ai_app.py
```

Open browser → **http://localhost:7860**

---

## What You Get

### Tab 1: 🔍 Wisdom Tooth Detection
Upload X-ray → AI analyzes → Bounding boxes drawn → Results displayed

### Tab 2: 💬 Multi-Model Chatbot
Ask question → 3 models respond in parallel → Compare answers

---

## File Overview

| File | Purpose | Lines |
|------|---------|-------|
| [dental_ai_app.py](backend/dental_ai_app.py) | Main Gradio app with 2 tabs | ~300 |
| [api_utils.py](backend/api_utils.py) | API calls (vision + chat) | ~240 |
| [image_utils.py](backend/image_utils.py) | Image processing & bounding boxes | ~180 |
| [test_example.py](backend/test_example.py) | Test API connectivity | ~100 |

---

## Architecture

```
User Input
    ↓
Gradio UI (dental_ai_app.py)
    ↓
┌───────────────┬──────────────────┐
│   Tab 1       │      Tab 2       │
│   Vision      │      Chat        │
└───────┬───────┴──────────┬───────┘
        ↓                  ↓
   api_utils.py       api_utils.py
        ↓                  ↓
  ┌─────┴─────┐      ┌────┴────┬─────────┐
  GPT-4V  Gemini     GPT-4o  Gemini  Groq
        ↓                  ↓ (parallel)
   image_utils.py
        ↓
   Annotated Image
```

---

## Key Features

✅ **Modular Design**: 3 separate files for UI, API, and image processing
✅ **Async Execution**: Parallel API calls using asyncio
✅ **Error Handling**: Try-except blocks with user-friendly messages
✅ **Type Hints**: Clear function signatures
✅ **Environment Variables**: Secure API key management
✅ **Documentation**: Comprehensive README files

---

## Testing Before Launch

```bash
# Test API connectivity
python test_example.py

# Should show:
# ✅ OpenAI API key found
# ✅ Groq API key found
# ✅ Google AI API key found
# ✅ All clients initialized
# ✅ Simple chat tests passed
```

---

## Customize It

### Change default model:
**File:** `dental_ai_app.py:152`
```python
value="Gemini Vision"  # Change from GPT-4o Vision
```

### Adjust response length:
**File:** `api_utils.py:125,155,185`
```python
max_tokens=1000  # Increase for longer responses
```

### Change port:
**File:** `dental_ai_app.py:285`
```python
server_port=8080  # Change from 7860
```

---

## API Cost Estimates (per 1000 requests)

| Model | Input Cost | Output Cost | Total (avg) |
|-------|-----------|-------------|-------------|
| GPT-4o | $2.50 | $10.00 | ~$6 |
| GPT-4o Vision | $5.00 | $15.00 | ~$10 |
| Gemini Flash | $0.075 | $0.30 | ~$0.20 |
| Gemini Vision | $0.075 | $0.30 | ~$0.20 |
| Groq Llama3 | FREE | FREE | FREE |

**Tip:** Start with Groq for chat and Gemini Vision for X-rays to minimize costs!

---

## Troubleshooting One-Liners

```bash
# Module not found
pip install -r requirements.txt

# Port in use
lsof -ti:7860 | xargs kill -9

# Check Python version (need 3.8+)
python3 --version

# Reinstall everything
rm -rf venv && ./setup.sh
```

---

## Next Steps

1. ✅ Run `python test_example.py` to verify APIs
2. ✅ Launch `python dental_ai_app.py`
3. ✅ Test with sample X-ray image
4. ✅ Try the chatbot with example questions
5. 📖 Read [DENTAL_AI_README.md](backend/DENTAL_AI_README.md) for advanced features

---

## Production Checklist

Before deploying:
- [ ] Add rate limiting (see Gradio docs)
- [ ] Implement user authentication
- [ ] Set up logging (Python logging module)
- [ ] Add response caching (Redis)
- [ ] Configure HTTPS
- [ ] Set up monitoring (Sentry, DataDog)
- [ ] Create backup strategy for analysis data
- [ ] Add CORS configuration if needed
- [ ] Implement API usage tracking
- [ ] Create admin dashboard

---

## Support

- 📖 Full docs: [backend/DENTAL_AI_README.md](backend/DENTAL_AI_README.md)
- 🐛 Issues: Check troubleshooting section in README
- 💡 Examples: See `test_example.py` for API usage patterns

**Happy coding!** 🦷✨
