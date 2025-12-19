# 🏗️ Architecture Documentation

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Dental AI Platform                       │
│                    (Gradio Web Interface)                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
   ┌────▼────┐         ┌────▼────┐
   │  Tab 1  │         │  Tab 2  │
   │ Vision  │         │  Chat   │
   └────┬────┘         └────┬────┘
        │                   │
        │                   │
┌───────▼────────┐    ┌─────▼──────────────────┐
│  Vision APIs   │    │   Chat APIs (Async)    │
│                │    │                        │
│  - GPT-4o-V    │    │  ┌──────────────────┐  │
│  - Gemini-V    │    │  │  OpenAI GPT-4o   │  │
└───────┬────────┘    │  ├──────────────────┤  │
        │             │  │  Google Gemini   │  │
┌───────▼────────┐    │  ├──────────────────┤  │
│ Image Utils    │    │  │  Groq Llama3     │  │
│                │    │  └──────────────────┘  │
│ - Parse JSON   │    │    (Parallel Exec)     │
│ - Draw Boxes   │    └────────────────────────┘
│ - Annotate     │
└────────────────┘
```

## Component Breakdown

### 1. Main Application (`dental_ai_app.py`)

**Purpose:** Gradio UI orchestration

```python
Responsibilities:
├── Define UI layout (tabs, buttons, inputs)
├── Handle user events (clicks, uploads)
├── Route requests to appropriate modules
└── Display results to user
```

**Key Functions:**
- `process_xray(image, model_choice)` → Processes X-ray uploads
- `chat_with_all_models(query)` → Handles multi-model chat

**Gradio Components Used:**
- `gr.Image()` - X-ray upload
- `gr.Textbox()` - Chat input
- `gr.Markdown()` - Response display
- `gr.Radio()` - Model selection
- `gr.Button()` - Action triggers
- `gr.Tabs()` - Tab organization

---

### 2. API Utilities (`api_utils.py`)

**Purpose:** Abstract API interactions

```python
Vision APIs:
├── analyze_xray_gpt4v()
│   ├── Encode image to base64
│   ├── Call OpenAI vision API
│   └── Return structured response
│
└── analyze_xray_gemini()
    ├── Convert PIL image
    ├── Call Gemini vision API
    └── Return structured response

Chat APIs (Async):
├── chat_openai_async()
│   └── Run in thread pool executor
│
├── chat_gemini_async()
│   └── Run in thread pool executor
│
├── chat_groq_async()
│   └── Run in thread pool executor
│
└── chat_all_models()
    ├── Create async tasks
    ├── asyncio.gather() - parallel execution
    └── Return all results
```

**Why Async?**
```
Sequential:  [GPT-4o: 5s] → [Gemini: 3s] → [Groq: 2s] = 10s total
Parallel:    [GPT-4o: 5s]
             [Gemini: 3s]  } = 5s total (max of all)
             [Groq: 2s]
```

**Error Handling:**
```python
try:
    response = api_call()
    return {"success": True, "response": data}
except Exception as e:
    return {"success": False, "error": str(e)}
```

---

### 3. Image Utilities (`image_utils.py`)

**Purpose:** Image processing & annotation

```python
Image Pipeline:
├── parse_vision_response()
│   ├── Extract JSON from markdown
│   ├── Handle code blocks
│   └── Parse to dict
│
├── draw_bounding_boxes()
│   ├── Convert % coords to pixels
│   ├── Draw rectangles (PIL)
│   ├── Add labels
│   └── Color code by position
│
└── create_side_by_side_comparison()
    ├── Resize images to match height
    ├── Combine horizontally
    └── Add labels
```

**Color Mapping:**
```python
{
    "upper-left": "#FF6B6B",    # Red
    "upper-right": "#4ECDC4",   # Teal
    "lower-left": "#FFE66D",    # Yellow
    "lower-right": "#95E1D3"    # Mint
}
```

**Coordinate System:**
```
Vision APIs return: [x_min%, y_min%, x_max%, y_max%]
Convert to pixels: coord * image_dimension

Example:
  Image: 1000x800 pixels
  Bbox: [0.6, 0.7, 0.85, 0.95]
  →     [600, 560, 850, 760] pixels
```

---

## Data Flow Diagrams

### Tab 1: Wisdom Tooth Detection Flow

```
User Action
    │
    ├─→ Upload Image (PNG/JPG)
    │       │
    │       ↓
    │   PIL.Image object
    │       │
    ├─→ Select Model (Radio)
    │       │
    ├─→ Click "Analyze"
    │       │
    │       ↓
    ├─→ process_xray()
            │
            ├─→ if GPT-4o Vision:
            │       │
            │       ├─→ encode_image_to_base64()
            │       │
            │       ├─→ analyze_xray_gpt4v()
            │       │       │
            │       │       ├─→ OpenAI API call
            │       │       │
            │       │       └─→ Return JSON response
            │       │
            │   or Gemini Vision:
            │       │
            │       ├─→ analyze_xray_gemini()
            │               │
            │               ├─→ Google Gemini API call
            │               │
            │               └─→ Return JSON response
            │
            ├─→ parse_vision_response()
            │       │
            │       ├─→ Extract JSON from markdown
            │       │
            │       └─→ Parse teeth_found[]
            │
            ├─→ draw_bounding_boxes()
            │       │
            │       ├─→ For each tooth:
            │       │       ├─→ Convert % to pixels
            │       │       ├─→ Draw rectangle
            │       │       └─→ Add label
            │       │
            │       └─→ Return annotated image
            │
            └─→ Format analysis text
                    │
                    └─→ Display results
```

### Tab 2: Multi-Model Chat Flow

```
User Action
    │
    ├─→ Type Question
    │       │
    ├─→ Click "Ask All Models"
    │       │
    │       ↓
    ├─→ chat_with_all_models()
            │
            ├─→ Create asyncio event loop
            │
            ├─→ chat_all_models() [ASYNC]
            │       │
            │       ├─→ Create 3 async tasks:
            │       │       │
            │       │       ├─→ Task 1: chat_openai_async()
            │       │       │       │
            │       │       │       └─→ GPT-4o API (in thread pool)
            │       │       │
            │       │       ├─→ Task 2: chat_gemini_async()
            │       │       │       │
            │       │       │       └─→ Gemini API (in thread pool)
            │       │       │
            │       │       └─→ Task 3: chat_groq_async()
            │       │               │
            │       │               └─→ Groq API (in thread pool)
            │       │
            │       ├─→ await asyncio.gather(*tasks)
            │       │       │
            │       │       └─→ Wait for ALL to complete
            │       │
            │       └─→ Return (result1, result2, result3)
            │
            ├─→ Format responses with ✅/❌
            │
            └─→ Display in 3 columns
```

---

## API Integration Details

### OpenAI GPT-4o Vision

**Endpoint:** `chat.completions.create`
**Input Format:**
```python
{
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "prompt"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,..."
                }}
            ]
        }
    ]
}
```

**Response:** JSON with text content

---

### Google Gemini Vision

**Endpoint:** `GenerativeModel.generate_content`
**Input Format:**
```python
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content([prompt_text, pil_image])
```

**Response:** `response.text` contains analysis

---

### Chat APIs (All)

**Common Pattern:**
```python
messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": query}
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=500,
    temperature=0.7
)

answer = response.choices[0].message.content
```

---

## Performance Optimization

### 1. Parallel API Calls

**Implementation:**
```python
async def chat_all_models():
    tasks = [
        chat_openai_async(query),
        chat_gemini_async(query),
        chat_groq_async(query)
    ]
    # All run concurrently
    results = await asyncio.gather(*tasks)
    return results
```

**Benefit:** 3x faster than sequential

---

### 2. Thread Pool for Sync APIs

**Problem:** OpenAI/Groq SDKs are synchronous
**Solution:** Run in executor

```python
async def chat_openai_async():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Default thread pool
        lambda: openai_client.chat.completions.create(...)
    )
    return result
```

---

### 3. Image Encoding Cache

**Future Enhancement:**
```python
@lru_cache(maxsize=128)
def encode_image_to_base64(image_hash):
    # Cache encoded images
    pass
```

---

## Error Handling Strategy

### Layer 1: API Call Level
```python
try:
    response = api_call()
    return {"success": True, "response": data}
except APIError as e:
    return {"success": False, "error": f"API Error: {e}"}
except Exception as e:
    return {"success": False, "error": f"Unexpected: {e}"}
```

### Layer 2: Processing Level
```python
if not result["success"]:
    return None, f"❌ Error: {result['error']}"
```

### Layer 3: UI Level
```python
# Gradio displays error message
# User sees friendly error without crash
```

---

## Security Considerations

### 1. API Key Management
```
✅ Load from .env file
✅ Never log API keys
✅ Never expose in error messages
✅ .gitignore prevents commits
```

### 2. Input Validation
```python
if not message.strip():
    return "Please enter a message"

if not image:
    return None, "Please upload image"
```

### 3. Rate Limiting (TODO)
```python
# Future enhancement
from gradio.utils import rate_limit

@rate_limit(max_calls=10, period=60)
def process_xray():
    pass
```

---

## Testing Strategy

### 1. Unit Tests (test_example.py)
```python
✓ Test API key loading
✓ Test client initialization
✓ Test simple API calls
```

### 2. Integration Tests (Manual)
```
✓ Upload image → verify annotation
✓ Ask question → verify 3 responses
✓ Test error handling (invalid image)
```

### 3. Performance Tests
```
✓ Measure parallel vs sequential
✓ Check response times
✓ Monitor API usage
```

---

## Deployment Architecture

### Local Development
```
User Browser
    ↓
localhost:7860 (Gradio)
    ↓
Local Python Process
    ↓
External APIs
```

### Production (Example)
```
Users
    ↓
HTTPS/SSL
    ↓
Nginx Reverse Proxy
    ↓
Gunicorn + Gradio
    ↓
Redis (Cache)
    ↓
External APIs
```

---

## Extension Points

### Add New Vision Model
1. Create `analyze_xray_newmodel()` in `api_utils.py`
2. Add to model selector in `dental_ai_app.py`
3. Update UI choices

### Add New Chat Model
1. Create `chat_newmodel_async()` in `api_utils.py`
2. Add to `chat_all_models()` tasks
3. Add output column in UI

### Add New Feature Tab
1. Create new tab in `dental_ai_app.py`
2. Add processing function
3. Connect UI components

---

## Dependencies Graph

```
dental_ai_app.py
    ├─→ gradio
    ├─→ dotenv
    ├─→ PIL
    ├─→ api_utils
    │       ├─→ openai
    │       ├─→ groq
    │       ├─→ google.generativeai
    │       └─→ asyncio
    └─→ image_utils
            ├─→ PIL
            ├─→ cv2
            ├─→ numpy
            └─→ json
```

---

## File Size & Complexity

| File | Lines | Functions | Complexity |
|------|-------|-----------|------------|
| dental_ai_app.py | ~300 | 3 | Medium |
| api_utils.py | ~240 | 9 | High |
| image_utils.py | ~180 | 5 | Medium |
| test_example.py | ~100 | 3 | Low |

**Total:** ~820 lines of production code

---

## Future Enhancements

1. **Response Caching** - Redis for repeated queries
2. **Batch Processing** - Multiple X-rays at once
3. **Export Reports** - PDF generation with results
4. **User Accounts** - Authentication & history
5. **Model Versioning** - A/B testing different models
6. **Analytics Dashboard** - Usage statistics
7. **Webhook Integration** - Connect to PACS systems
8. **Mobile App** - React Native wrapper

---

**Architecture designed for:**
- ✅ Modularity
- ✅ Scalability
- ✅ Maintainability
- ✅ Extensibility
- ✅ Performance
