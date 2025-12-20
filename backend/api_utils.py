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
        import traceback
        error_details = traceback.format_exc()

        print("=" * 60)
        print("❌ ERROR in analyze_xray_gpt4v")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print(f"Has Image: {image is not None}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 60)

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
        import traceback
        error_details = traceback.format_exc()

        print("=" * 60)
        print("❌ ERROR in chat_openai_async")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 60)

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
        import traceback
        error_details = traceback.format_exc()

        print("=" * 60)
        print("❌ ERROR in chat_gemini_async")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 60)

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
        import traceback
        error_details = traceback.format_exc()

        print("=" * 60)
        print("❌ ERROR in chat_groq_async")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 60)

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
        messages: List of conversation messages with role/content (may include image fields to filter)
        model_name: "gpt4", "gemini", or "groq"
        openai_client: OpenAI client instance
        groq_client: Groq client instance

    Returns:
        Dict with model response
    """
    try:
        loop = asyncio.get_event_loop()

        # Clean messages - remove image fields for text-only models
        # Also ensure content is always a string, not dict or other type
        clean_messages = []
        for msg in messages:
            # Make sure we have role and content
            if "role" not in msg or "content" not in msg:
                continue

            # Extract content as string
            content = msg["content"]
            if isinstance(content, dict):
                # If content is a dict, extract text part
                content = str(content)
            elif not isinstance(content, str):
                content = str(content)

            clean_msg = {
                "role": msg["role"],
                "content": content
            }
            clean_messages.append(clean_msg)

        if model_name == "gpt4":
            # Debug: Print message structure
            print(f"[DEBUG] Sending {len(clean_messages)} messages to GPT-4")
            for i, msg in enumerate(clean_messages):
                print(f"  [{i}] role={msg['role']}, content_type={type(msg['content'])}, preview={str(msg['content'])[:60]}")

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
            def sync_gemini_context_call():
                model = genai.GenerativeModel('gemini-1.5-flash')

                # Convert messages to Gemini format
                chat_history = []
                for msg in clean_messages[1:]:  # Skip system prompt for now
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
        import traceback
        error_details = traceback.format_exc()

        # Log error details
        print("=" * 60)
        print(f"❌ ERROR in chat_with_context_async ({model_name})")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print(f"Messages count: {len(messages) if messages else 0}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 60)

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
        messages: Conversation history (may include PIL Images that need conversion)
        image: PIL Image to analyze (current image)
        model_name: "gpt4-vision" or "gemini-vision"
        openai_client: OpenAI client

    Returns:
        Dict with vision analysis
    """
    try:
        loop = asyncio.get_event_loop()

        if model_name == "gpt4-vision":
            # Convert PIL Images in context to base64
            vision_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    vision_messages.append(msg)
                elif msg['role'] == 'user':
                    # Check if this message has an image
                    if msg.get('image') and isinstance(msg['image'], Image.Image):
                        # Convert to vision message format
                        base64_img = encode_image_to_base64(msg['image'])
                        vision_messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": msg['content']},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                            ]
                        })
                    else:
                        # Text-only message
                        vision_messages.append({"role": "user", "content": msg['content']})
                elif msg['role'] == 'assistant':
                    vision_messages.append(msg)

            # Add current image if provided
            if image:
                base64_image = encode_image_to_base64(image)
                # The last message should be the user's current message
                # We need to replace or append the image to it
                if vision_messages and vision_messages[-1]['role'] == 'user':
                    # Update last user message to include image
                    last_msg = vision_messages[-1]
                    if isinstance(last_msg['content'], str):
                        vision_messages[-1] = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": last_msg['content']},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]
                        }

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

                # Build prompt from conversation context
                prompt_parts = []
                for msg in messages:
                    if msg['role'] == 'user':
                        prompt_parts.append(f"User: {msg['content']}")
                    elif msg['role'] == 'assistant':
                        prompt_parts.append(f"Assistant: {msg['content']}")

                # Add current message
                prompt = "\n".join(prompt_parts) + "\n\nAnalyze this dental X-ray and identify wisdom teeth. Provide their positions and any notable conditions."

                # Gemini accepts PIL Image directly
                response = model.generate_content([prompt, image])
                return response.text

            response_text = await loop.run_in_executor(None, sync_gemini_vision)
            return {
                "model": "gemini-vision",
                "response": response_text,
                "success": True
            }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        # Log error details
        print("=" * 60)
        print(f"❌ ERROR in vision_with_context_async ({model_name})")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print(f"Has Image: {image is not None}")
        print(f"Messages count: {len(messages) if messages else 0}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 60)

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
        message: User's text message (for reference, already in conversation_context)
        image: Optional image upload (current image for vision models)
        conversation_context: Full conversation history (already includes current message)
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
            # conversation_context already includes the current message (added in dental_ai_unified.py)
            tasks.append(chat_with_context_async(
                conversation_context,
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
