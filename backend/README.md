# Multi-Model AI Chatbot Backend

A Gradio-based chatbot application that supports multiple AI models from OpenAI, Groq, and Google.

## Features

- 🤖 **Multiple AI Models**: Switch between 7 different AI models
  - OpenAI: GPT-4, GPT-3.5 Turbo
  - Groq: Llama 3.1 (70B & 8B), Mixtral 8x7B
  - Google: Gemini Pro, Gemini 1.5 Flash

- 💬 **Conversation History**: Maintains context across messages
- 🎨 **Beautiful UI**: Modern, responsive Gradio interface
- ⚡ **Fast Response**: Optimized API calls with async support
- 🔄 **Model Switching**: Change models mid-conversation

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**

   Make sure your `.env` file contains:
   ```env
   OPEN_AI_API_KEY=your_openai_key_here
   GROQ_AI_API_KEY=your_groq_key_here
   GOOGLE_AI_API_KEY=your_google_key_here
   ```

## Usage

**Run the application:**
```bash
python app.py
```

The chatbot will be available at: `http://localhost:7860`

## API Keys

You need API keys from:

1. **OpenAI**: https://platform.openai.com/api-keys
2. **Groq**: https://console.groq.com/keys
3. **Google AI**: https://makersuite.google.com/app/apikey

## Model Comparison

| Model | Provider | Speed | Capability | Best For |
|-------|----------|-------|------------|----------|
| GPT-4 | OpenAI | Medium | Highest | Complex reasoning |
| GPT-3.5 Turbo | OpenAI | Fast | Good | General tasks |
| Llama 3.1 70B | Groq | Very Fast | High | Long conversations |
| Llama 3.1 8B | Groq | Fastest | Good | Quick responses |
| Mixtral 8x7B | Groq | Very Fast | High | Technical tasks |
| Gemini Pro | Google | Fast | High | Multimodal tasks |
| Gemini 1.5 Flash | Google | Fastest | Good | Quick queries |

## Project Structure

```
backend/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (API keys)
└── README.md          # This file
```

## Troubleshooting

**Error: Invalid API key**
- Check that your API keys are correctly set in the `.env` file
- Ensure there are no extra spaces or quotes around the keys

**Error: Module not found**
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.8 or higher

**Slow responses**
- Try switching to faster models (Groq Llama 8B or Gemini Flash)
- Check your internet connection

## Advanced Usage

### Customize the System Prompt

Edit the system message in `app.py`:
```python
messages = [{"role": "system", "content": "Your custom prompt here"}]
```

### Adjust Temperature

Modify the `temperature` parameter for more creative (higher) or focused (lower) responses:
```python
temperature=0.7  # Default: 0.7, Range: 0.0 - 2.0
```

### Change Port

Modify the launch parameters:
```python
demo.launch(server_port=8080)  # Change from 7860 to 8080
```

## License

MIT License
