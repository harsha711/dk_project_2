"""
API utility functions for vision and chat models
"""
import os
import base64
import asyncio
from io import BytesIO
from typing import Dict, Tuple, Optional
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

# ============ CONTEXT-AWARE CHAT FUNCTIONS (FOR UNIFIED CHATBOT) ============

async def chat_with_context_async(
    messages: list,
    model_name: str,
    openai_client: OpenAI = None,
    groq_client: Groq = None
) -> Dict:
    """
    Chat with conversation context for any model
    
    Args:
        messages: List of conversation messages with role/content
        model_name: "gpt4", "gemini", or "groq"
        openai_client: OpenAI client instance
        groq_client: Groq client instance
    
    Returns:
        Dict with model response
    """
    try:
        loop = asyncio.get_event_loop()
        
        if model_name == "gpt4":
            response = await loop.run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7
                )
            )
            return {
                "model": "gpt4",
                "response": response.choices[0].message.content,
                "success": True
            }
        
        elif model_name == "gemini":
            def sync_gemini_context_call():
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Convert messages to Gemini format
                chat_history = []
                for msg in messages[1:]:  # Skip system prompt for now
                    role = "user" if msg["role"] == "user" else "model"
                    chat_history.append({"role": role, "parts": [msg["content"]]})
                
                # Start chat with history (excluding last message)
                if len(chat_history) > 1:
                    chat = model.start_chat(history=chat_history[:-1])
                    response = chat.send_message(chat_history[-1]["parts"][0])
                else:
                    # First message
                    response = model.generate_content(chat_history[0]["parts"][0])
                
                return response.text
            
            response_text = await loop.run_in_executor(None, sync_gemini_context_call)
            return {
                "model": "gemini",
                "response": response_text,
                "success": True
            }
        
        elif model_name == "groq":
            response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7
                )
            )
            return {
                "model": "groq",
                "response": response.choices[0].message.content,
                "success": True
            }
        
    except Exception as e:
        return {
            "model": model_name,
            "response": f"Error: {str(e)}",
            "success": False
        }


async def vision_with_context_async(
    messages: list,
    image: Image.Image,
    model_name: str,
    openai_client: OpenAI = None
) -> Dict:
    """
    Vision analysis with conversation context
    
    Args:
        messages: Conversation history
        image: PIL Image to analyze
        model_name: "gpt4-vision" or "gemini-vision"
        openai_client: OpenAI client
    
    Returns:
        Dict with vision analysis
    """
    try:
        loop = asyncio.get_event_loop()
        
        if model_name == "gpt4-vision":
            base64_image = encode_image_to_base64(image)
            
            # Add image to last user message
            vision_messages = messages.copy()
            vision_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this dental X-ray for wisdom teeth."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            })
            
            response = await loop.run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=vision_messages,
                    max_tokens=1000,
                    temperature=0.3
                )
            )
            
            return {
                "model": "gpt4-vision",
                "response": response.choices[0].message.content,
                "success": True
            }
        
        elif model_name == "gemini-vision":
            def sync_gemini_vision():
                # Use gemini-pro-vision for vision tasks
                model = genai.GenerativeModel('gemini-pro-vision')
                prompt = "Analyze this dental X-ray and identify wisdom teeth. Provide their positions and any notable conditions."
                response = model.generate_content([prompt, image])
                return response.text

            response_text = await loop.run_in_executor(None, sync_gemini_vision)
            return {
                "model": "gemini-vision",
                "response": response_text,
                "success": True
            }
    
    except Exception as e:
        return {
            "model": model_name,
            "response": f"Error: {str(e)}",
            "success": False
        }


async def multimodal_chat_async(
    message: str,
    image: Optional[Image.Image],
    conversation_context: list,
    models: list,
    openai_client: OpenAI,
    groq_client: Groq
) -> Dict:
    """
    Unified multimodal chat - handles both text and vision
    
    Args:
        message: User's text message
        image: Optional image upload
        conversation_context: Full conversation history
        models: List of model names to query
        openai_client: OpenAI client
        groq_client: Groq client
    
    Returns:
        Dict mapping model names to responses
    """
    tasks = []
    
    for model in models:
        if model in ["gpt4-vision", "gemini-vision"]:
            # Vision models
            tasks.append(vision_with_context_async(
                conversation_context,
                image,
                model,
                openai_client
            ))
        else:
            # Text chat models
            # Add current message to context
            full_context = conversation_context + [{"role": "user", "content": message}]
            tasks.append(chat_with_context_async(
                full_context,
                model,
                openai_client,
                groq_client
            ))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    response_dict = {}
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            response_dict[models[i]] = {
                "response": f"Error: {str(result)}",
                "success": False
            }
        else:
            response_dict[result["model"]] = result["response"]
    
    return response_dict
