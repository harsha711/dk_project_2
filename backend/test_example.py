"""
Example test file to verify API connectivity
Run this before launching the full Gradio app to test your API keys
"""
import os
from dotenv import load_dotenv
from api_utils import init_clients, analyze_xray_gemini
from PIL import Image
import io

# Load environment variables
load_dotenv()

def test_api_keys():
    """Test if API keys are loaded"""
    print("🔍 Testing API Keys...\n")

    openai_key = os.getenv("OPEN_AI_API_KEY")
    groq_key = os.getenv("GROQ_AI_API_KEY")
    google_key = os.getenv("GOOGLE_AI_API_KEY")

    if openai_key and openai_key.startswith("sk-"):
        print("✅ OpenAI API key found")
    else:
        print("❌ OpenAI API key missing or invalid")

    if groq_key and groq_key.startswith("gsk_"):
        print("✅ Groq API key found")
    else:
        print("❌ Groq API key missing or invalid")

    if google_key and google_key.startswith("AIza"):
        print("✅ Google AI API key found")
    else:
        print("❌ Google AI API key missing or invalid")

    print()


def test_client_initialization():
    """Test if API clients can be initialized"""
    print("🔧 Testing Client Initialization...\n")

    try:
        openai_client, groq_client = init_clients()
        print("✅ OpenAI client initialized")
        print("✅ Groq client initialized")
        print("✅ Google AI configured via genai\n")
        return openai_client, groq_client
    except Exception as e:
        print(f"❌ Client initialization failed: {e}\n")
        return None, None


def test_simple_chat(openai_client, groq_client):
    """Test simple chat functionality"""
    print("💬 Testing Chat APIs...\n")

    test_query = "Say 'Hello' in one word."

    # Test OpenAI
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": test_query}],
            max_tokens=10
        )
        print(f"✅ OpenAI GPT-4o: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ OpenAI failed: {e}")

    # Test Groq
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": test_query}],
            max_tokens=10
        )
        print(f"✅ Groq Llama3: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Groq failed: {e}")

    # Test Gemini
    try:
        from google import generativeai as genai
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(test_query)
        print(f"✅ Google Gemini: {response.text}")
    except Exception as e:
        print(f"❌ Gemini failed: {e}")

    print()


if __name__ == "__main__":
    print("="*50)
    print("🧪 Dental AI Platform - API Test Suite")
    print("="*50)
    print()

    # Run tests
    test_api_keys()

    openai_client, groq_client = test_client_initialization()

    if openai_client and groq_client:
        test_simple_chat(openai_client, groq_client)

    print("="*50)
    print("✅ Testing complete!")
    print("If all tests passed, you're ready to run: python dental_ai_app.py")
    print("="*50)
