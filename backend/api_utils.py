"""
API utility functions for vision and chat models
"""
import os
import base64
import asyncio
from io import BytesIO
from typing import Dict, Tuple
from openai import OpenAI
from groq import Groq
from google import generativeai as genai
from PIL import Image


# Initialize clients
def init_clients():
    """Initialize API clients with keys from environment"""
    openai_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_AI_API_KEY"))
    genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))

    return openai_client, groq_client


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# ============ VISION API FUNCTIONS ============

def analyze_xray_gpt4v(image: Image.Image, openai_client: OpenAI) -> Dict:
    """
    Analyze dental X-ray using GPT-4V
    Returns: dict with location info and coordinates
    """
    try:
        base64_image = encode_image_to_base64(image)

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this dental X-ray image and identify wisdom teeth locations.

For each wisdom tooth found, provide:
1. Position (upper-left, upper-right, lower-left, lower-right)
2. Bounding box coordinates as percentages of image dimensions (x_min, y_min, x_max, y_max)
3. Brief description of tooth condition

Format your response as JSON:
{
    "teeth_found": [
        {
            "position": "lower-right",
            "bbox": [0.6, 0.7, 0.85, 0.95],
            "description": "Impacted wisdom tooth"
        }
    ],
    "summary": "Brief overall summary"
}

If no wisdom teeth are visible, return empty teeth_found list."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )

        return {
            "success": True,
            "model": "GPT-4o Vision",
            "response": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "success": False,
            "model": "GPT-4o Vision",
            "error": str(e)
        }


def analyze_xray_gemini(image: Image.Image) -> Dict:
    """
    Analyze dental X-ray using Gemini Vision
    Returns: dict with location info and coordinates
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = """Analyze this dental X-ray image and identify wisdom teeth locations.

For each wisdom tooth found, provide:
1. Position (upper-left, upper-right, lower-left, lower-right)
2. Bounding box coordinates as percentages of image dimensions (x_min, y_min, x_max, y_max)
3. Brief description of tooth condition

Format your response as JSON:
{
    "teeth_found": [
        {
            "position": "lower-right",
            "bbox": [0.6, 0.7, 0.85, 0.95],
            "description": "Impacted wisdom tooth"
        }
    ],
    "summary": "Brief overall summary"
}

If no wisdom teeth are visible, return empty teeth_found list."""

        response = model.generate_content([prompt, image])

        return {
            "success": True,
            "model": "Gemini Vision",
            "response": response.text
        }

    except Exception as e:
        return {
            "success": False,
            "model": "Gemini Vision",
            "error": str(e)
        }


# ============ CHAT API FUNCTIONS ============

async def chat_openai_async(query: str, openai_client: OpenAI) -> Dict:
    """Async chat with OpenAI GPT-4o"""
    try:
        # Run sync OpenAI call in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": query}],
                max_tokens=500,
                temperature=0.7
            )
        )

        return {
            "model": "OpenAI GPT-4o",
            "response": response.choices[0].message.content,
            "success": True
        }
    except Exception as e:
        return {
            "model": "OpenAI GPT-4o",
            "response": f"Error: {str(e)}",
            "success": False
        }


async def chat_gemini_async(query: str) -> Dict:
    """Async chat with Google Gemini"""
    try:
        loop = asyncio.get_event_loop()

        def sync_gemini_call():
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(query)
            return response.text

        response_text = await loop.run_in_executor(None, sync_gemini_call)

        return {
            "model": "Google Gemini",
            "response": response_text,
            "success": True
        }
    except Exception as e:
        return {
            "model": "Google Gemini",
            "response": f"Error: {str(e)}",
            "success": False
        }


async def chat_groq_async(query: str, groq_client: Groq) -> Dict:
    """Async chat with Groq Llama3"""
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": query}],
                max_tokens=500,
                temperature=0.7
            )
        )

        return {
            "model": "Groq Llama3",
            "response": response.choices[0].message.content,
            "success": True
        }
    except Exception as e:
        return {
            "model": "Groq Llama3",
            "response": f"Error: {str(e)}",
            "success": False
        }


async def chat_all_models(query: str, openai_client: OpenAI, groq_client: Groq) -> Tuple[Dict, Dict, Dict]:
    """
    Send query to all 3 models in parallel
    Returns: (openai_result, gemini_result, groq_result)
    """
    tasks = [
        chat_openai_async(query, openai_client),
        chat_gemini_async(query),
        chat_groq_async(query, groq_client)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            model_name = ["OpenAI GPT-4o", "Google Gemini", "Groq Llama3"][i]
            processed_results.append({
                "model": model_name,
                "response": f"Error: {str(result)}",
                "success": False
            })
        else:
            processed_results.append(result)

    return tuple(processed_results)
