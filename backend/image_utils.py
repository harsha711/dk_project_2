"""
Image processing utilities for dental X-ray analysis
"""
import json
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional


def parse_vision_response(response_text: str) -> Dict:
    """
    Parse JSON response from vision models
    Returns parsed dict or error dict
    """
    try:
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            # Try to find JSON object in the response
            # Look for { ... } pattern
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end + 1].strip()
            else:
                json_str = response_text.strip()

        parsed = json.loads(json_str)

        # Log what was parsed
        print(f"[PARSE DEBUG] Successfully parsed JSON with {len(parsed.get('teeth_found', []))} teeth")

        return parsed

    except json.JSONDecodeError as e:
        print(f"[PARSE ERROR] Failed to parse JSON: {str(e)}")
        print(f"[PARSE ERROR] Response preview: {response_text[:300]}...")
        return {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_response": response_text,
            "teeth_found": []  # Return empty list to prevent errors downstream
        }


def draw_bounding_boxes(image: Image.Image, detections: List[Dict], show_confidence: bool = True) -> Image.Image:
    """
    Draw bounding boxes on image based on detection results

    Supports both YOLO and LLM detection formats:
    - YOLO format: includes 'confidence' and 'class_name' fields
    - LLM format: traditional 'position', 'bbox', 'description'

    Args:
        image: PIL Image
        detections: List of detection dicts with 'position', 'bbox', 'description'
                   (and optionally 'confidence', 'class_name' for YOLO detections)
        show_confidence: Whether to show confidence scores (default: True)

    Returns:
        Annotated PIL Image
    """
    # Create a copy to avoid modifying original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Image dimensions
    width, height = img_copy.size

    # Color map for different positions
    color_map = {
        "upper-left": "#FF6B6B",    # Red
        "upper-right": "#4ECDC4",   # Teal
        "lower-left": "#FFE66D",    # Yellow
        "lower-right": "#95E1D3"    # Mint
    }

    # Class-specific colors for YOLO detections
    class_color_map = {
        "Impacted": "#FF6B6B",      # Red
        "Caries": "#FF9F40",        # Orange
        "Deep Caries": "#FF4444",   # Dark Red
        "Periapical Lesion": "#9F40FF",  # Purple
        "Crown": "#4ECDC4",         # Teal
        "Filling": "#95E1D3",       # Mint
        "Implant": "#FFE66D",       # Yellow
    }

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for idx, detection in enumerate(detections):
        position = detection.get("position", "unknown")
        bbox = detection.get("bbox", [0.1, 0.1, 0.3, 0.3])  # Default bbox
        description = detection.get("description", "Wisdom tooth")

        # YOLO-specific fields
        confidence = detection.get("confidence", None)
        class_name = detection.get("class_name", None)

        # Convert percentage coordinates to pixels
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)

        # Determine color: use class-specific color if available, otherwise position-based
        if class_name and class_name in class_color_map:
            color = class_color_map[class_name]
        else:
            color = color_map.get(position.lower(), "#FF0000")

        # Draw rectangle with thick outline for better visibility
        # Draw outer rectangle first (slightly larger) for better visibility
        draw.rectangle(
            [(x_min - 1, y_min - 1), (x_max + 1, y_max + 1)],
            outline=(0, 0, 0),  # Black outline for contrast
            width=2
        )
        # Draw main colored rectangle
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=color,
            width=5  # Thick outline for clear visibility
        )

        # Create label based on detection type
        if class_name and confidence is not None and show_confidence:
            # YOLO detection - show class and confidence
            label = f"{class_name} ({confidence:.0%})"
        elif class_name:
            # YOLO detection without confidence
            label = f"{class_name}"
        else:
            # LLM detection - traditional format
            label = f"{position}: {description}"

        # Use textbbox for better text positioning
        text_bbox = draw.textbbox((x_min, y_min - 25), label, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle for text
        draw.rectangle(
            [(x_min, y_min - text_height - 30), (x_min + text_width + 10, y_min - 5)],
            fill=color,
            outline=color
        )

        # Draw text
        draw.text(
            (x_min + 5, y_min - text_height - 25),
            label,
            fill="black",
            font=font_small
        )

    return img_copy


def resize_image_for_chat(image: Image.Image, max_width: int = 500, max_height: int = 400) -> Image.Image:
    """
    Resize image to fit in chat display while maintaining aspect ratio
    
    Args:
        image: PIL Image to resize
        max_width: Maximum width in pixels (default: 500 - reduced for better chat display)
        max_height: Maximum height in pixels (default: 400 - reduced for better chat display)
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    # Calculate scaling factor to fit within max dimensions
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale = min(width_ratio, height_ratio, 1.0)  # Don't upscale
    
    # Only resize if image is larger than max dimensions
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        print(f"  📐 Resized image from {width}x{height} to {new_width}x{new_height} for chat display")
        return resized
    
    return image


