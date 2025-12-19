"""
Image processing utilities for dental X-ray analysis
"""
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple


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
            json_str = response_text.strip()

        parsed = json.loads(json_str)
        return parsed

    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_response": response_text
        }


def draw_bounding_boxes(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Draw bounding boxes on image based on detection results

    Args:
        image: PIL Image
        detections: List of detection dicts with 'position', 'bbox', 'description'

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

        # Convert percentage coordinates to pixels
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)

        # Get color for this position
        color = color_map.get(position.lower(), "#FF0000")

        # Draw rectangle
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=color,
            width=4
        )

        # Draw label background
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


def create_side_by_side_comparison(original: Image.Image, annotated: Image.Image) -> Image.Image:
    """
    Create side-by-side comparison of original and annotated images
    """
    # Ensure both images are the same height
    max_height = max(original.size[1], annotated.size[1])

    # Resize if needed (maintain aspect ratio)
    if original.size[1] != max_height:
        aspect_ratio = original.size[0] / original.size[1]
        original = original.resize((int(max_height * aspect_ratio), max_height), Image.LANCZOS)

    if annotated.size[1] != max_height:
        aspect_ratio = annotated.size[0] / annotated.size[1]
        annotated = annotated.resize((int(max_height * aspect_ratio), max_height), Image.LANCZOS)

    # Create new image with combined width
    total_width = original.size[0] + annotated.size[1] + 20  # 20px gap
    combined = Image.new('RGB', (total_width, max_height), color='white')

    # Paste images
    combined.paste(original, (0, 0))
    combined.paste(annotated, (original.size[0] + 20, 0))

    # Add labels
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), "Original X-Ray", fill="white", font=font)
    draw.text((original.size[0] + 30, 10), "AI Analysis", fill="white", font=font)

    return combined


def add_summary_overlay(image: Image.Image, summary_text: str) -> Image.Image:
    """
    Add summary text overlay at the bottom of image
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Create semi-transparent overlay for text background
    width, height = img_copy.size
    overlay_height = 80

    # Draw semi-transparent rectangle at bottom
    draw.rectangle(
        [(0, height - overlay_height), (width, height)],
        fill=(0, 0, 0, 180)
    )

    # Draw summary text
    draw.text(
        (10, height - overlay_height + 10),
        f"Summary: {summary_text}",
        fill="white",
        font=font
    )

    return img_copy
