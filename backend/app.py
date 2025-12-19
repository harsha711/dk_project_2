import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from google import generativeai as genai

# Load environment variables
load_dotenv()

# Initialize AI clients
openai_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_AI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))

# Model configurations
MODELS = {
    "OpenAI GPT-4": {"provider": "openai", "model": "gpt-4"},
    "OpenAI GPT-3.5 Turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
    "Groq Llama 3.1 (70B)": {"provider": "groq", "model": "llama-3.1-70b-versatile"},
    "Groq Llama 3.1 (8B)": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "Groq Mixtral (8x7B)": {"provider": "groq", "model": "mixtral-8x7b-32768"},
    "Google Gemini Pro": {"provider": "google", "model": "gemini-pro"},
    "Google Gemini 1.5 Flash": {"provider": "google", "model": "gemini-1.5-flash"}
}


def chat_with_openai(message, history, model_name):
    """Chat with OpenAI models"""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Add conversation history
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    # Add current message
    messages.append({"role": "user", "content": message})

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )

    return response.choices[0].message.content


def chat_with_groq(message, history, model_name):
    """Chat with Groq models"""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Add conversation history
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    # Add current message
    messages.append({"role": "user", "content": message})

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )

    return response.choices[0].message.content


def chat_with_google(message, history, model_name):
    """Chat with Google Gemini models"""
    model = genai.GenerativeModel(model_name)

    # Convert history to Gemini format
    chat_history = []
    for human, assistant in history:
        chat_history.append({"role": "user", "parts": [human]})
        chat_history.append({"role": "model", "parts": [assistant]})

    chat = model.start_chat(history=chat_history)
    response = chat.send_message(message)

    return response.text


def respond(message, history, selected_model):
    """Main chat function that routes to the appropriate AI provider"""
    if not message.strip():
        return ""

    if selected_model not in MODELS:
        return "Please select a valid model from the dropdown."

    model_config = MODELS[selected_model]
    provider = model_config["provider"]
    model_name = model_config["model"]

    try:
        if provider == "openai":
            response = chat_with_openai(message, history, model_name)
        elif provider == "groq":
            response = chat_with_groq(message, history, model_name)
        elif provider == "google":
            response = chat_with_google(message, history, model_name)
        else:
            response = "Invalid provider selected."

        return response

    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check your API keys and try again."


# Custom CSS for better styling
custom_css = """
#component-0 {
    max-width: 900px;
    margin: auto;
    padding-top: 1.5rem;
}

#chatbot {
    height: 600px;
}

.model-info {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div class="model-info">
            <h1 style="margin: 0; font-size: 2.5em;">🤖 Multi-Model AI Chatbot</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.1em;">
                Chat with multiple AI models: OpenAI, Groq, and Google Gemini
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Chat",
                height=600,
                bubble_full_width=False
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    scale=4,
                    lines=2
                )

            with gr.Row():
                submit = gr.Button("Send 🚀", variant="primary", scale=1)
                clear = gr.Button("Clear 🗑️", scale=1)

        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Groq Llama 3.1 (8B)",
                label="🎯 Select Model",
                info="Choose your AI model"
            )

            gr.Markdown("""
            ### 📊 Available Models

            **OpenAI:**
            - GPT-4 (Most capable)
            - GPT-3.5 Turbo (Fast)

            **Groq:**
            - Llama 3.1 70B (Powerful)
            - Llama 3.1 8B (Lightning fast)
            - Mixtral 8x7B (Great reasoning)

            **Google:**
            - Gemini Pro (Advanced)
            - Gemini 1.5 Flash (Quick)

            ### 💡 Tips
            - Switch models mid-conversation
            - Each model has unique strengths
            - Groq models are fastest
            """)

    # Event handlers
    msg.submit(respond, [msg, chatbot, model_selector], chatbot).then(
        lambda: "", None, msg
    )

    submit.click(respond, [msg, chatbot, model_selector], chatbot).then(
        lambda: "", None, msg
    )

    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
