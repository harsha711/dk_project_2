"""
API utility functions for vision and chat models

CLEAN VERSION:
- YOLO for accurate bounding box detection
- GPT-4 and Groq for text-based chat
- All vision models removed (were hallucinating coordinates)
"""
import os
import base64
import asyncio
from io import BytesIO
from typing import Dict, Optional
from openai import OpenAI
from groq import Groq
from PIL import Image
import numpy as np
from ultralytics import YOLO


# Initialize clients
def init_clients():
    """Initialize API clients with keys from environment"""
    openai_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_AI_API_KEY"))

    return openai_client, groq_client


# ============ YOLO MODEL INITIALIZATION ============
_yolo_model = None

YOLO_IMPACTED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dental_impacted.pt")
YOLO_FALLBACK_PATH = "yolov8n.pt"

def get_yolo_model():
    """Get or initialize YOLO model (lazy loading)"""
    global _yolo_model

    if _yolo_model is None:
        try:
            if os.path.exists(YOLO_IMPACTED_MODEL_PATH):
                print(f"✅ Loading DENTEX model with Impacted class from {YOLO_IMPACTED_MODEL_PATH}")
                _yolo_model = YOLO(YOLO_IMPACTED_MODEL_PATH)
                print(f"   Model classes: {list(_yolo_model.names.values())}")
            else:
                print(f"⚠️ No dental models found")
                print(f"📥 Using base YOLOv8n model (NOT trained on dental X-rays)")
                _yolo_model = YOLO(YOLO_FALLBACK_PATH)
        except Exception as e:
            print(f"❌ Error loading YOLO model: {str(e)}")
            print(f"📥 Downloading base YOLOv8n model...")
            _yolo_model = YOLO(YOLO_FALLBACK_PATH)

    return _yolo_model


def detect_teeth_yolo(image: Image.Image, conf_threshold: float = 0.25) -> Dict:
    """
    Detect teeth in dental X-ray using YOLOv8 model

    Args:
        image: PIL Image of dental X-ray
        conf_threshold: Confidence threshold for detections (default: 0.25)

    Returns:
        Dict with detected teeth information
    """
    try:
        model = get_yolo_model()
        img_array = np.array(image)
        results = model(img_array, conf=conf_threshold, verbose=False)

        img_height, img_width = img_array.shape[:2]
        teeth_found = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                bbox_normalized = [
                    float(x1 / img_width),
                    float(y1 / img_height),
                    float(x2 / img_width),
                    float(y2 / img_height)
                ]

                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"

                center_x = (bbox_normalized[0] + bbox_normalized[2]) / 2
                center_y = (bbox_normalized[1] + bbox_normalized[3]) / 2

                if center_y < 0.5:
                    position = "upper-left" if center_x < 0.5 else "upper-right"
                else:
                    position = "lower-left" if center_x < 0.5 else "lower-right"

                description = f"{class_name} (confidence: {confidence:.2f})"

                teeth_found.append({
                    "position": position,
                    "bbox": bbox_normalized,
                    "confidence": confidence,
                    "class_name": class_name,
                    "description": description
                })

        teeth_found.sort(key=lambda x: x['confidence'], reverse=True)

        summary = f"Detected {len(teeth_found)} teeth/dental features using YOLO" if teeth_found else "No teeth detected in the X-ray image"

        print(f"  🦷 YOLO detected {len(teeth_found)} teeth/features")

        return {
            "success": True,
            "model": "YOLOv8 Dental Detection",
            "teeth_found": teeth_found,
            "summary": summary
        }

    except Exception as e:
        print(f"❌ ERROR in detect_teeth_yolo: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "model": "YOLOv8 Dental Detection",
            "teeth_found": [],
            "summary": f"Error during YOLO detection: {str(e)}"
        }


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# ============ TEXT CHAT FUNCTIONS ============

async def chat_with_context_async(
    messages: list,
    model_name: str,
    openai_client: OpenAI = None,
    groq_client: Groq = None
) -> Dict:
    """Chat with conversation context (text only)"""
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

        print(f"[{model_name.upper()} CONTEXT] Sending {len(clean_messages)} messages")

        # Debug: Show last message to verify YOLO context is included
        if clean_messages and len(clean_messages) > 0:
            last_msg = clean_messages[-1]
            preview = last_msg.get('content', '')[:200] if isinstance(last_msg.get('content'), str) else str(last_msg.get('content'))[:200]
            print(f"  Last message preview: {preview}...")

        if model_name == "gpt4":
            response = await loop.run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o-mini",
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


async def multimodal_chat_async(
    message: str,
    image: Optional[Image.Image],  # Unused - kept for API compatibility
    conversation_context: list,
    models: list,
    openai_client: OpenAI,
    groq_client: Groq
) -> Dict:
    """
    Text-only chat with multiple models
    NOTE: Image parameter unused - text models cannot see images
    YOLO handles all image detection separately
    """
    tasks = []

    # Call text models (gpt4, groq only - gemini removed)
    for model in models:
        if model in ["gpt4", "groq"]:
            tasks.append(chat_with_context_async(
                conversation_context,
                model,
                openai_client,
                groq_client
            ))

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks, return_exceptions=True)

    response_dict = {}
    valid_models = [m for m in models if m in ["gpt4", "groq"]]

    for i, result in enumerate(results):
        if i < len(valid_models):
            if isinstance(result, Exception):
                response_dict[valid_models[i]] = f"Error: {str(result)}"
            else:
                response_dict[result.get("model", valid_models[i])] = result["response"]

    return response_dict
