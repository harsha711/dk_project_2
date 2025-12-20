"""
API utility functions for vision and chat models
FIXED VERSION - Correct Groq Vision model names
"""
import os
import base64
import asyncio
import json
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


# ============ GROQ VISION MODELS (UPDATED) ============
# As of Dec 2024, Groq supports these vision models:
# - llama-3.2-11b-vision-preview (smaller, faster)
# - llama-3.2-90b-vision-preview (larger, more accurate)
# - meta-llama/llama-4-scout-17b-16e-instruct (newer Llama 4)
# - meta-llama/llama-4-maverick-17b-128e-instruct (newer Llama 4)

GROQ_VISION_MODELS = [
    "llama-3.2-11b-vision-preview",      # Most reliable for vision
    "llama-3.2-90b-vision-preview",      # Larger but may have availability issues
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 Scout
]


def analyze_xray_groq_vision(image: Image.Image, groq_client: Groq) -> Dict:
    """
    Analyze dental X-ray using Groq Llama Vision
    Returns: dict with location info and coordinates
    """
    try:
        base64_image = encode_image_to_base64(image)
        
        vision_prompt = """[ACADEMIC RESEARCH PROJECT - EDUCATIONAL PURPOSES ONLY]

This is a computer vision training exercise for an academic dental AI project. Analyze this sample dental X-ray image and identify wisdom teeth locations.

For each wisdom tooth found, provide:
1. Position (upper-left, upper-right, lower-left, lower-right)
2. Bounding box coordinates as percentages of image dimensions (x_min, y_min, x_max, y_max)
3. Brief description of tooth condition

Your response MUST be valid JSON in this exact format:
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

If no wisdom teeth are visible, return: {"teeth_found": [], "summary": "No wisdom teeth detected"}

IMPORTANT: Return ONLY the JSON object, no additional text or explanations."""

        # Try each model in order until one works
        last_error = None
        for model_name in GROQ_VISION_MODELS:
            try:
                print(f"  🔄 Trying Groq vision model: {model_name}")
                
                response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
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
                
                print(f"  ✅ Successfully used Groq model: {model_name}")
                return {
                    "success": True,
                    "model": f"Groq Vision ({model_name})",
                    "response": response.choices[0].message.content
                }
                
            except Exception as model_error:
                print(f"  ⚠️ Model {model_name} failed: {str(model_error)[:100]}")
                last_error = model_error
                continue
        
        # All models failed
        return {
            "success": False,
            "model": "Groq Vision",
            "error": f"All Groq vision models failed. Last error: {str(last_error)}"
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        print("=" * 60)
        print("❌ ERROR in analyze_xray_groq_vision")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 60)
        
        return {
            "success": False,
            "model": "Groq Vision",
            "error": str(e)
        }


# ============ VISION API FUNCTIONS ============
# NOTE: GPT-4o Vision removed - refuses medical image analysis requests
# Available vision models: Gemini Vision, Groq Llama Vision


def analyze_xray_gemini(image: Image.Image) -> Dict:
    """
    Analyze dental X-ray using Gemini Vision
    Returns: dict with location info and coordinates
    """
    try:
        # Use gemini-2.0-flash (current stable model as of Dec 2024)
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = """[ACADEMIC RESEARCH PROJECT - EDUCATIONAL PURPOSES ONLY]

This is a computer vision training exercise for an academic dental AI project. Analyze this sample dental X-ray image for educational demonstration purposes to identify wisdom teeth locations as a training example.

For each wisdom tooth found in this training sample, provide:
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

DISCLAIMER: This is for academic research and training purposes only. Not for clinical diagnosis.

If no wisdom teeth are visible, return empty teeth_found list."""

        response = model.generate_content([prompt, image])

        return {
            "success": True,
            "model": "Gemini Vision",
            "response": response.text
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        print("=" * 60)
        print("❌ ERROR in analyze_xray_gemini")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print(f"Has Image: {image is not None}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 60)

        return {
            "success": False,
            "model": "Gemini Vision",
            "error": str(e)
        }


# ============ CHAT API FUNCTIONS ============

async def chat_openai_async(query: str, openai_client: OpenAI) -> Dict:
    """Async chat with OpenAI GPT-4o"""
    try:
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
            model = genai.GenerativeModel('gemini-2.0-flash')
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
                model="llama-3.3-70b-versatile",
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
    """
    try:
        loop = asyncio.get_event_loop()

        # Clean messages - remove image fields for text-only models
        clean_messages = []
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                continue
            content = msg["content"]
            if isinstance(content, dict):
                content = str(content)
            elif not isinstance(content, str):
                content = str(content)
            clean_msg = {"role": msg["role"], "content": content}
            clean_messages.append(clean_msg)

        if model_name == "gpt4":
            response = await loop.run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=clean_messages,
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
            def sync_gemini():
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt_parts = []
                for msg in clean_messages:
                    if msg['role'] == 'system':
                        prompt_parts.append(f"System: {msg['content']}")
                    elif msg['role'] == 'user':
                        prompt_parts.append(f"User: {msg['content']}")
                    elif msg['role'] == 'assistant':
                        prompt_parts.append(f"Assistant: {msg['content']}")
                full_prompt = "\n".join(prompt_parts)
                response = model.generate_content(full_prompt)
                return response.text

            response_text = await loop.run_in_executor(None, sync_gemini)
            return {
                "model": "gemini",
                "response": response_text,
                "success": True
            }
        
        elif model_name == "groq":
            response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=clean_messages,
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


# ============ VISION WITH CONTEXT (FOR UNIFIED CHATBOT) ============

async def vision_with_context_async(
    messages: list,
    image: Image.Image,
    model_name: str,
    openai_client: OpenAI = None,
    groq_client: Groq = None
) -> Dict:
    """
    Vision analysis with conversation context
    """
    try:
        loop = asyncio.get_event_loop()
        
        if image is None:
            return {
                "model": model_name,
                "response": "No image provided for vision analysis",
                "success": False
            }

        if model_name == "gemini-vision":
            def sync_gemini_vision():
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt_parts = []
                for msg in messages:
                    if msg['role'] == 'user':
                        prompt_parts.append(f"User: {msg['content']}")
                    elif msg['role'] == 'assistant':
                        prompt_parts.append(f"Assistant: {msg['content']}")

                prompt = "\n".join(prompt_parts) + """

[ACADEMIC RESEARCH - EDUCATIONAL SAMPLE]

Analyze this sample dental X-ray for educational demonstration purposes. Identify wisdom teeth locations and describe their positions and conditions as a training example.

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

This is for academic research only, not clinical diagnosis.
If no wisdom teeth are visible, return empty teeth_found list."""

                response = model.generate_content([prompt, image])
                return response.text

            response_text = await loop.run_in_executor(None, sync_gemini_vision)
            return {
                "model": "gemini-vision",
                "response": response_text,
                "success": True
            }

        elif model_name == "groq-vision":
            def sync_groq_vision():
                base64_image = encode_image_to_base64(image)
                
                user_message = messages[-1]['content'] if messages and messages[-1].get('role') == 'user' else 'Analyze this dental X-ray'
                
                vision_prompt = f"""{user_message}

[ACADEMIC RESEARCH PROJECT - EDUCATIONAL PURPOSES ONLY]

Analyze this dental X-ray image and identify wisdom teeth locations.

For each wisdom tooth found, provide:
1. Position (upper-left, upper-right, lower-left, lower-right)
2. Bounding box coordinates as percentages (x_min, y_min, x_max, y_max)
3. Brief description of tooth condition

Your response MUST be valid JSON:
{{
    "teeth_found": [
        {{
            "position": "lower-right",
            "bbox": [0.6, 0.7, 0.85, 0.95],
            "description": "Impacted wisdom tooth"
        }}
    ],
    "summary": "Brief overall summary"
}}

If no wisdom teeth visible: {{"teeth_found": [], "summary": "No wisdom teeth detected"}}

Return ONLY the JSON object."""

                # Try each vision model
                for model_id in GROQ_VISION_MODELS:
                    try:
                        print(f"  🔄 Trying Groq vision: {model_id}")
                        response = groq_client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": vision_prompt},
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
                        print(f"  ✅ Success with: {model_id}")
                        return response.choices[0].message.content
                    except Exception as e:
                        print(f"  ⚠️ {model_id} failed: {str(e)[:80]}")
                        continue
                
                # All failed - return error message as JSON
                return json.dumps({
                    "teeth_found": [],
                    "summary": "Groq vision models unavailable. Please use Gemini Vision instead."
                })

            response_text = await loop.run_in_executor(None, sync_groq_vision)
            return {
                "model": "groq-vision",
                "response": response_text,
                "success": True
            }

    except Exception as e:
        import traceback
        print(f"❌ ERROR in vision_with_context_async: {str(e)}")
        traceback.print_exc()
        
        return {
            "model": model_name,
            "response": f"Error: {str(e)}",
            "success": False
        }


# ============ UNIFIED MULTIMODAL CHAT ============

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
    """
    tasks = []
    
    for model in models:
        if model in ["gemini-vision", "groq-vision"]:
            tasks.append(vision_with_context_async(
                conversation_context,
                image,
                model,
                openai_client,
                groq_client
            ))
        else:
            tasks.append(chat_with_context_async(
                conversation_context,
                model,
                openai_client,
                groq_client
            ))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
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