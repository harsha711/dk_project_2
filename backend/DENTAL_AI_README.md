# 🦷 Dental AI Platform

A comprehensive Gradio-based application featuring AI-powered dental X-ray analysis and multi-model chatbot capabilities.

## 🌟 Features

### Tab 1: Wisdom Tooth Detection
- Upload dental X-ray images
- AI-powered wisdom tooth detection using GPT-4o Vision or Gemini Vision
- Automatic bounding box annotation on detected teeth
- Detailed analysis with position, description, and coordinates
- Side-by-side comparison view

### Tab 2: Multi-Model Chatbot
- Parallel queries to 3 leading AI models:
  - **OpenAI GPT-4o** - Most capable reasoning
  - **Google Gemini 1.5 Flash** - Fast and efficient
  - **Groq Llama3 70B** - Ultra-fast inference
- Side-by-side response comparison
- Async/concurrent API calls for optimal performance
- Example questions included

## 📁 Project Structure

```
backend/
├── dental_ai_app.py      # Main Gradio application
├── api_utils.py          # API client functions (vision + chat)
├── image_utils.py        # Image processing & annotation utilities
├── requirements.txt      # Python dependencies
├── .env                  # API keys (keep secret!)
├── setup.sh             # Installation script
├── run.sh               # Quick launch script
└── DENTAL_AI_README.md  # This file
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd backend
./setup.sh
source venv/bin/activate
python dental_ai_app.py
```

### Option 2: Manual Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python dental_ai_app.py
```

### Option 3: Quick Run (if already set up)

```bash
cd backend
./run.sh
```

Then open your browser to: **http://localhost:7860**

## 🔑 API Keys Setup

Make sure your `.env` file contains valid API keys:

```env
OPEN_AI_API_KEY=sk-...
GROQ_AI_API_KEY=gsk_...
GOOGLE_AI_API_KEY=AIza...
```

### Where to Get API Keys:

1. **OpenAI**: https://platform.openai.com/api-keys
   - Sign up and create a new API key
   - Requires payment method for GPT-4o access

2. **Groq**: https://console.groq.com/keys
   - Free tier available with generous limits
   - Extremely fast inference

3. **Google AI**: https://makersuite.google.com/app/apikey
   - Free tier available
   - Also accessible via Google Cloud Console

## 💻 Dependencies

```
gradio==4.44.0          # Web UI framework
openai==1.54.0          # OpenAI API client
groq==0.11.0            # Groq API client
google-generativeai     # Google Gemini API
python-dotenv           # Environment variable management
pillow                  # Image processing
opencv-python           # Computer vision utilities
numpy                   # Numerical operations
aiohttp                 # Async HTTP requests
```

## 🎯 Usage Guide

### Tab 1: Wisdom Tooth Detection

1. **Upload X-Ray Image**
   - Click "Upload Dental X-Ray"
   - Select a dental X-ray image (PNG, JPG, etc.)
   - Panoramic X-rays work best

2. **Select AI Model**
   - **GPT-4o Vision**: Better accuracy, slower
   - **Gemini Vision**: Fast, good accuracy

3. **Analyze**
   - Click "Analyze X-Ray" button
   - Wait for AI analysis (5-15 seconds)
   - View annotated image with bounding boxes
   - Read detailed analysis in the text panel

4. **Interpretation**
   - Bounding boxes show detected wisdom teeth locations
   - Colors indicate different quadrants:
     - Red: Upper left
     - Teal: Upper right
     - Yellow: Lower left
     - Mint: Lower right

### Tab 2: Multi-Model Chatbot

1. **Enter Question**
   - Type your question in the text box
   - Or select an example question

2. **Submit**
   - Click "Ask All Models" or press Enter
   - All 3 models process in parallel (3-8 seconds)

3. **Compare Responses**
   - View responses side by side
   - ✅ indicates successful response
   - ❌ indicates error (check API key/quota)

4. **Clear**
   - Click "Clear" button to reset

## 🏗️ Architecture

### Modular Design

**dental_ai_app.py**
- Main Gradio interface
- Event handlers and UI logic
- Tab organization

**api_utils.py**
- Vision API functions:
  - `analyze_xray_gpt4v()` - GPT-4o Vision analysis
  - `analyze_xray_gemini()` - Gemini Vision analysis
- Chat API functions:
  - `chat_openai_async()` - Async OpenAI chat
  - `chat_gemini_async()` - Async Gemini chat
  - `chat_groq_async()` - Async Groq chat
  - `chat_all_models()` - Parallel execution

**image_utils.py**
- `parse_vision_response()` - JSON parsing from AI responses
- `draw_bounding_boxes()` - Image annotation with PIL
- `create_side_by_side_comparison()` - Image comparison layout
- `add_summary_overlay()` - Text overlay on images

### Async Architecture

The chatbot uses Python's `asyncio` for concurrent API calls:

```python
# Parallel execution example
tasks = [
    chat_openai_async(query, client),
    chat_gemini_async(query),
    chat_groq_async(query, client)
]
results = await asyncio.gather(*tasks)
```

This reduces total response time from ~15s (sequential) to ~5s (parallel).

## 🔧 Customization

### Change Model Defaults

In `dental_ai_app.py`, modify:

```python
model_selector = gr.Radio(
    choices=["GPT-4o Vision", "Gemini Vision"],
    value="Gemini Vision",  # Change default here
    ...
)
```

### Adjust Response Length

In `api_utils.py`, modify `max_tokens`:

```python
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": query}],
    max_tokens=1000,  # Increase for longer responses
    temperature=0.7
)
```

### Add More Models

To add a new chat model, update `api_utils.py`:

```python
async def chat_new_model_async(query: str) -> Dict:
    # Your implementation
    pass

# Update chat_all_models() to include it
async def chat_all_models(...):
    tasks = [
        chat_openai_async(...),
        chat_gemini_async(...),
        chat_groq_async(...),
        chat_new_model_async(...)  # Add here
    ]
    ...
```

### Modify Bounding Box Colors

In `image_utils.py`, edit `color_map`:

```python
color_map = {
    "upper-left": "#YOUR_COLOR",
    "upper-right": "#YOUR_COLOR",
    ...
}
```

### Change Port

In `dental_ai_app.py`:

```python
demo.launch(
    server_port=8080,  # Change from 7860
    ...
)
```

## 🐛 Troubleshooting

### Error: "Invalid API key"
- Check `.env` file for correct API keys
- Ensure no extra spaces or quotes
- Verify keys are active on provider dashboards

### Error: "Rate limit exceeded"
- You've hit API quota limits
- Wait a few minutes or upgrade your plan
- Groq has generous free tier limits

### Error: "Module not found"
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated
- Check Python version (need 3.8+)

### Slow Response Times
- Normal for GPT-4o Vision (10-15s)
- Groq is fastest for chat (~2s)
- Check internet connection
- Consider using Gemini Flash for speed

### Bounding Boxes Not Appearing
- AI may not detect wisdom teeth in image
- Check that image is a dental X-ray
- Try different AI model
- Review raw JSON response for debugging

### Import Errors
```bash
# Fix: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Port Already in Use
```bash
# Fix: Kill process on port 7860
lsof -ti:7860 | xargs kill -9

# Or change port in dental_ai_app.py
```

## 📊 Model Comparison

| Feature | GPT-4o Vision | Gemini Vision |
|---------|--------------|---------------|
| **Accuracy** | Excellent | Very Good |
| **Speed** | 10-15s | 5-8s |
| **Cost** | Higher | Lower/Free |
| **Detail** | Very detailed | Good detail |
| **Best For** | Critical analysis | Quick checks |

| Chat Model | Speed | Quality | Cost | Free Tier |
|------------|-------|---------|------|-----------|
| **GPT-4o** | Medium | Excellent | $$$ | Limited |
| **Gemini Flash** | Fast | Very Good | $ | Yes |
| **Groq Llama3** | Very Fast | Good | Free | Generous |

## 🔒 Security Notes

- **NEVER commit `.env` file** (already in `.gitignore`)
- Rotate API keys regularly
- Use environment variables in production
- Implement rate limiting for public deployments
- Consider API key encryption for sensitive apps

## 🚀 Performance Optimization

### For Vision APIs:
- Resize large images before upload (max 2000px)
- Use Gemini for faster processing
- Cache results for repeated images

### For Chat APIs:
- Parallel execution already implemented
- Groq provides fastest inference
- Reduce `max_tokens` for quicker responses
- Implement response caching for common queries

## 📝 Example Use Cases

### Dental Practice
- Pre-consultation wisdom tooth screening
- Patient education with visual annotations
- Treatment planning assistance

### Educational
- Dental student training
- AI model comparison demonstrations
- Medical imaging analysis examples

### Research
- Benchmark vision model accuracy
- Compare AI responses across models
- Dataset annotation tool

## 🤝 Contributing

To extend this platform:

1. Add new models in `api_utils.py`
2. Create new image processing in `image_utils.py`
3. Update UI in `dental_ai_app.py`
4. Test with various X-ray images
5. Update documentation

## 📄 License

MIT License - Feel free to use and modify

## 🙏 Acknowledgments

- **Gradio**: Amazing UI framework
- **OpenAI**: GPT-4o Vision API
- **Google**: Gemini AI
- **Groq**: Ultra-fast inference
- **Pillow/OpenCV**: Image processing

---

**Built with ❤️ for dental AI applications**
