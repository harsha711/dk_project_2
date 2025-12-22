"""
Image processing utilities for dental X-ray analysis
"""
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List
import numpy as np

# Try to import OpenCV for CLAHE enhancement
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️ OpenCV not available. CLAHE enhancement will be disabled.")


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
        text_bbox = draw.textbbox((0, 0), label, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Padding for text box
        text_padding = 10
        label_box_width = text_width + text_padding * 2
        label_box_height = text_height + text_padding * 2

        # Determine label position to avoid cutoff
        # Try above first, then below, then inside if needed
        label_x = x_min
        label_y = y_min - label_box_height - 5  # Above box
        
        # Check if label would be cut off at top
        if label_y < 0:
            # Place below box instead
            label_y = y_max + 5
            # Check if label would be cut off at bottom
            if label_y + label_box_height > height:
                # Place inside box at top
                label_y = y_min + 5
        
        # Check if label would be cut off on right edge
        if label_x + label_box_width > width:
            # Move label left to fit
            label_x = width - label_box_width - 5
            # If still doesn't fit, place it at the left edge
            if label_x < 0:
                label_x = 5
        
        # Check if label would be cut off on left edge
        if label_x < 0:
            label_x = 5

        # Draw background rectangle for text
        draw.rectangle(
            [(label_x, label_y), (label_x + label_box_width, label_y + label_box_height)],
            fill=color,
            outline=(0, 0, 0),  # Black outline for better visibility
            width=1
        )

        # Draw text
        draw.text(
            (label_x + text_padding, label_y + text_padding),
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


def apply_clahe(image: Image.Image, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance medical X-ray images.

    CLAHE is particularly effective for revealing subtle details in dental X-rays that might be
    missed by the naked eye, such as:
    - Hidden impacted wisdom teeth
    - Small cavities
    - Early-stage bone loss
    - Root canal issues

    Args:
        image: PIL Image to enhance
        clip_limit: Contrast limiting threshold (default: 3.0)
                   Higher values = more contrast (range: 1.0-5.0 recommended)
        tile_grid_size: Size of grid for local histogram equalization (default: (8, 8))
                       Smaller = more local adaptation, larger = smoother

    Returns:
        Enhanced PIL Image with CLAHE applied

    Note:
        If OpenCV is not available, returns the original image unchanged.
    """
    if not OPENCV_AVAILABLE:
        print("⚠️ CLAHE enhancement skipped: OpenCV not installed")
        print("   Install with: pip install opencv-python-headless")
        return image

    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Check if image is grayscale or RGB
        if len(img_array.shape) == 2:
            # Grayscale image - apply CLAHE directly
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(img_array)

        else:
            # RGB image - convert to LAB color space
            # LAB separates luminance (L) from color (A, B)
            # This allows us to enhance contrast without affecting colors
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

            # Split into L, A, B channels
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to L-channel only (luminance)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_channel_enhanced = clahe.apply(l_channel)

            # Merge channels back
            lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])

            # Convert back to RGB
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # Convert back to PIL Image
        enhanced_image = Image.fromarray(enhanced)

        print(f"✅ CLAHE applied: clipLimit={clip_limit}, tileGridSize={tile_grid_size}")
        return enhanced_image

    except Exception as e:
        print(f"❌ Error applying CLAHE: {str(e)}")
        print("   Returning original image")
        return image


