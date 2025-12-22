"""
API utility functions for vision and chat models

CLEAN VERSION:
- YOLO for accurate bounding box detection
- GPT-4 and Groq for text-based chat
- All vision models removed (were hallucinating coordinates)
"""
import os
import asyncio
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


def detect_teeth_yolo(image: Image.Image, conf_threshold: float = 0.35, iou_threshold: float = 0.4) -> Dict:
    """
    Detect teeth in dental X-ray using YOLOv8 model

    Args:
        image: PIL Image of dental X-ray
        conf_threshold: Base confidence threshold for detections (default: 0.35 for better precision)
        iou_threshold: NMS IoU threshold (default: 0.4, more aggressive to reduce overlaps)

    Returns:
        Dict with detected teeth information
    """
    # Class-specific confidence thresholds to reduce false positives
    CLASS_CONFIDENCE_THRESHOLDS = {
        "Impacted": 0.25,           # Lower threshold to catch more impacted teeth
        "impacted": 0.25,           # Case-insensitive variant
        "Impacted Tooth": 0.25,     # Roboflow model class name
        "impacted tooth": 0.25,     # Case-insensitive
        "Cavity": 0.30,
        "Caries": 0.30,
        "caries": 0.30,
        "Deep Caries": 0.30,
        "deep caries": 0.30,
        "Fillings": 0.30,
        "Implant": 0.30,
    }

    try:
        model = get_yolo_model()
        img_array = np.array(image)

        print(f"🔍 YOLO Detection Debug:")
        print(f"  - Image size: {img_array.shape[:2]}")
        print(f"  - Base confidence threshold: {conf_threshold}")
        print(f"  - IoU threshold: {iou_threshold}")
        print(f"  - Model classes: {list(model.names.values()) if hasattr(model, 'names') else 'N/A'}")
        print(f"  - Class-specific thresholds: {CLASS_CONFIDENCE_THRESHOLDS}")

        # Use very low base threshold to catch all detections, then filter by class-specific thresholds
        results = model(img_array, conf=0.15, iou=iou_threshold, verbose=False)

        img_height, img_width = img_array.shape[:2]
        teeth_found = []
        filtered_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                print(f"  ⚠️ No boxes detected in this result")
                continue

            print(f"  ✅ Found {len(boxes)} raw detections (before filtering)")

            for idx, box in enumerate(boxes):
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

                # Apply class-specific confidence threshold
                min_confidence = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, conf_threshold)

                # Additional spatial filtering for "Impacted" detections
                skip_detection = False
                skip_reason = ""

                if "impacted" in class_name.lower():
                    # Impacted wisdom teeth are typically at the edges/back of the jaw
                    # Relaxed threshold: x < 0.30 (left third) or x > 0.70 (right third)
                    # This allows detection further into the jaw while still filtering middle teeth
                    if center_y >= 0.5:  # Lower jaw
                        if not (center_x < 0.30 or center_x > 0.70):
                            skip_detection = True
                            skip_reason = f"Impacted in middle of jaw (x={center_x:.2f}, needs x<0.30 or x>0.70)"
                    else:  # Upper jaw
                        if not (center_x < 0.30 or center_x > 0.70):
                            skip_detection = True
                            skip_reason = f"Impacted in middle of upper jaw (x={center_x:.2f}, needs x<0.30 or x>0.70)"

                # Check confidence threshold
                if confidence < min_confidence:
                    skip_detection = True
                    skip_reason = f"Confidence {confidence:.3f} below threshold {min_confidence:.3f}"

                # Detailed debug for each detection
                print(f"    [{idx+1}] {class_name} @ {position}")
                print(f"        Confidence: {confidence:.3f} (min required: {min_confidence:.3f})")
                print(f"        Center: ({center_x:.3f}, {center_y:.3f})")
                print(f"        BBox: x1={bbox_normalized[0]:.3f}, y1={bbox_normalized[1]:.3f}, x2={bbox_normalized[2]:.3f}, y2={bbox_normalized[3]:.3f}")

                if skip_detection:
                    print(f"        ❌ FILTERED OUT: {skip_reason}")
                    filtered_count += 1
                    continue
                else:
                    print(f"        ✅ ACCEPTED")

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

        print(f"  🦷 YOLO final results: {len(teeth_found)} accepted, {filtered_count} filtered out")

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

        # Debug: Show system prompt and last message to verify YOLO context is included
        if clean_messages and len(clean_messages) > 0:
            # Check for system prompt
            system_prompt_found = False
            for msg in clean_messages:
                if msg.get('role') == 'system':
                    system_prompt_found = True
                    sys_preview = msg.get('content', '')[:150] if isinstance(msg.get('content'), str) else str(msg.get('content'))[:150]
                    print(f"  ✅ System prompt found: {sys_preview}...")
                    break
            if not system_prompt_found:
                print(f"  ⚠️ No system prompt found in messages")
            
            # Show last message (should contain YOLO context for follow-ups)
            last_msg = clean_messages[-1]
            preview = last_msg.get('content', '')[:300] if isinstance(last_msg.get('content'), str) else str(last_msg.get('content'))[:300]
            print(f"  Last message preview: {preview}...")
            
            # Check if YOLO context is present
            last_content = last_msg.get('content', '') if isinstance(last_msg.get('content'), str) else str(last_msg.get('content'))
            if 'YOLO' in last_content or 'detection' in last_content.lower() or 'X-RAY ANALYSIS CONTEXT' in last_content:
                print(f"  ✅ YOLO context detected in last message")
            else:
                print(f"  ⚠️ YOLO context NOT detected in last message")

        if model_name == "gpt4":
            response = await loop.run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=clean_messages,
                    max_tokens=2000,  # Increased for complete responses
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
                    max_tokens=2000,  # Increased for complete responses
                    temperature=0.7
                )
            )
            return {
                "model": "groq",
                "response": response.choices[0].message.content,
                "success": True
            }

        elif model_name == "qwen":
            # Use Qwen 3 32B as third model
            response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    model="qwen/qwen3-32b",  # Qwen 3 32B
                    messages=clean_messages,
                    max_tokens=2000,  # Increased for complete responses
                    temperature=0.7
                )
            )
            return {
                "model": "qwen",
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

    # Call text models (gpt4, groq, qwen)
    for model in models:
        if model in ["gpt4", "groq", "qwen"]:
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
    valid_models = [m for m in models if m in ["gpt4", "groq", "qwen"]]

    for i, result in enumerate(results):
        if i < len(valid_models):
            if isinstance(result, Exception):
                response_dict[valid_models[i]] = f"Error: {str(result)}"
            else:
                response_dict[result.get("model", valid_models[i])] = result["response"]

    return response_dict
