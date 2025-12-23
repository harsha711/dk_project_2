"""
Annotation Playground - Test Your Diagnostic Skills
Simple click-based annotation using only standard Gradio 6.x components
Uses filepaths instead of PIL objects for Gradio display (fixes corruption issues)
"""
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional
import random
import tempfile
import os
from api_utils import detect_teeth_yolo
from image_utils import draw_bounding_boxes
from dataset_utils import TeethDatasetManager
import gradio as gr


def save_image_for_gradio(pil_image):
    """Save PIL image to temp file and return path for Gradio display"""
    if pil_image is None:
        return None
    
    # Create temp file with unique name
    temp_dir = tempfile.gettempdir()
    import uuid
    temp_path = os.path.join(temp_dir, f"dental_temp_{uuid.uuid4().hex[:8]}.png")
    
    # DEBUG: Check image before conversion
    import numpy as np
    arr_before = np.array(pil_image)
    print(f"[DEBUG save_image_for_gradio] Before conversion:")
    print(f"  - Mode: {pil_image.mode}")
    print(f"  - Shape: {arr_before.shape}")
    print(f"  - Min/Max: {arr_before.min()} / {arr_before.max()}")
    print(f"  - Unique values: {len(np.unique(arr_before))}")
    
    # Ensure RGB mode - FIXED: Proper grayscale to RGB conversion
    if pil_image.mode == 'L':
        # Grayscale to RGB - preserve full range, don't threshold
        # Method: Convert each grayscale pixel to RGB by duplicating the value
        arr = np.array(pil_image)  # Shape: (H, W)
        if len(arr.shape) == 2:
            # Stack to create RGB: (H, W, 3)
            arr_rgb = np.stack([arr, arr, arr], axis=-1)
            pil_image = Image.fromarray(arr_rgb.astype(np.uint8), mode='RGB')
        else:
            # Fallback to PIL convert
            pil_image = pil_image.convert('RGB')
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # DEBUG: Check image after conversion
    arr_after = np.array(pil_image)
    print(f"[DEBUG save_image_for_gradio] After conversion:")
    print(f"  - Mode: {pil_image.mode}")
    print(f"  - Shape: {arr_after.shape}")
    print(f"  - Min/Max: {arr_after.min()} / {arr_after.max()}")
    print(f"  - Unique values: {len(np.unique(arr_after))}")
    
    # Save as PNG
    pil_image.save(temp_path, format='PNG')
    
    # DEBUG: Verify saved image
    verify_img = Image.open(temp_path)
    verify_arr = np.array(verify_img)
    print(f"[DEBUG save_image_for_gradio] Saved file verification:")
    print(f"  - Mode: {verify_img.mode}")
    print(f"  - Min/Max: {verify_arr.min()} / {verify_arr.max()}")
    print(f"  - Unique values: {len(np.unique(verify_arr))}")
    
    return temp_path


def load_uploaded_image(image_path, state: Dict):
    """Load user-uploaded image for annotation practice"""
    if image_path is None:
        return None, None, state, "❌ Please upload an image first.", "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    # Load from filepath
    try:
        image = Image.open(image_path)
        image.load()  # Force load
    except Exception as e:
        return None, None, state, f"❌ Error loading image: {str(e)}", "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    # Convert to RGB if needed - FIXED: Proper grayscale to RGB conversion
    if image.mode == 'L':
        # Grayscale to RGB - preserve full range, don't threshold
        import numpy as np
        arr = np.array(image)  # Shape: (H, W)
        if len(arr.shape) == 2:
            # Stack to create RGB: (H, W, 3)
            arr_rgb = np.stack([arr, arr, arr], axis=-1)
            image = Image.fromarray(arr_rgb.astype(np.uint8), mode='RGB')
        else:
            image = image.convert('RGB')
    elif image.mode not in ("RGB", "L"):
        try:
            image = image.convert("RGB")
        except Exception as e:
            return None, None, state, f"❌ Error converting image: {str(e)}", "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    # Reset state for new image (store PIL in state)
    state["original_image"] = image.copy()
    state["display_image"] = image.copy()
    state["user_clicks"] = []
    
    avg_score_text = f"**Stats:** Attempts: {state['attempts']} | Avg Score: {state['total_score'] / state['attempts']:.0f}%" if state['attempts'] > 0 else "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    # Return filepath for Gradio
    filepath = save_image_for_gradio(image)
    
    return filepath, None, state, "*Image loaded! Click on dental issues you spot.*", avg_score_text


def load_random_playground_image(state: Dict, dataset_manager: TeethDatasetManager):
    """Load random X-ray for annotation practice (from dataset - may be corrupted)"""
    # Try to load from dataset, but show warning if dataset is corrupted
    if not dataset_manager.loaded:
        dataset_manager.load_metadata()
    
    if dataset_manager.total_samples == 0:
        return None, None, state, "❌ Dataset not available. Please upload an image using the uploader above.", "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    # Get random sample
    index = random.randint(0, dataset_manager.total_samples - 1)
    result = dataset_manager.get_sample(index)
    
    if not result.get("success"):
        return None, None, state, f"❌ Error loading sample: {result.get('error', 'Unknown error')}\n\n💡 **Tip:** Try uploading your own image using the uploader above.", "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    image = result['image']
    
    # DEBUG: Print image info
    print(f"\n[DEBUG] Original image from dataset:")
    print(f"  - Type: {type(image)}")
    print(f"  - Mode: {image.mode}")
    print(f"  - Size: {image.size}")
    
    import numpy as np
    arr = np.array(image)
    print(f"  - Array shape: {arr.shape}")
    print(f"  - Dtype: {arr.dtype}")
    print(f"  - Min/Max: {arr.min()} / {arr.max()}")
    print(f"  - Mean: {arr.mean():.1f}")
    print(f"  - Unique values: {len(np.unique(arr))}")
    
    # Save original for comparison
    try:
        image.save("/tmp/debug_original.png")
        print(f"  - Saved to /tmp/debug_original.png")
    except Exception as e:
        print(f"  - Failed to save debug image: {e}")
    
    # Preserve original image - only convert if absolutely necessary
    # The image should already be processed and in RGB mode from dataset_utils
    if not isinstance(image, Image.Image):
        return None, None, state, f"❌ Invalid image type: {type(image)}", "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    # Convert to RGB if needed - FIXED: Proper grayscale to RGB conversion
    if image.mode == 'L':
        # Grayscale to RGB - preserve full range, don't threshold
        import numpy as np
        arr = np.array(image)  # Shape: (H, W)
        if len(arr.shape) == 2:
            # Stack to create RGB: (H, W, 3)
            arr_rgb = np.stack([arr, arr, arr], axis=-1)
            image = Image.fromarray(arr_rgb.astype(np.uint8), mode='RGB')
            print(f"[PLAYGROUND] Converted grayscale to RGB (preserved full range)")
        else:
            image = image.convert('RGB')
            print(f"[PLAYGROUND] Converted grayscale to RGB (fallback)")
    elif image.mode not in ("RGB", "L"):
        try:
            # Convert non-RGB modes to RGB for display
            image = image.convert("RGB")
            print(f"[PLAYGROUND] Converted image from {result['image'].mode} to RGB")
        except Exception as e:
            print(f"[ERROR] Failed to convert image mode {image.mode}: {e}")
            return None, None, state, f"❌ Error converting image: {str(e)}", "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    # Reset state for new image (store PIL in state)
    state["original_image"] = image.copy()  # Store unmarked original (never modified)
    state["display_image"] = image.copy()  # Display image (with markers)
    state["current_index"] = index
    state["user_clicks"] = []
    
    avg_score_text = f"**Stats:** Attempts: {state['attempts']} | Avg Score: {state['total_score'] / state['attempts']:.0f}%" if state['attempts'] > 0 else "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    # Return filepath for Gradio
    filepath = save_image_for_gradio(image)
    
    return filepath, None, state, f"*Image #{index} loaded. Click on dental issues!*", avg_score_text


def handle_image_click(state: Dict, evt: gr.SelectData):
    """Handle click on image and draw a marker using Gradio's .select() event"""
    if state is None or state.get("original_image") is None:
        return None, state, "Load an image first!"
    
    # Get click coordinates from SelectData event
    # evt.index can be a tuple (x, y) or a list [x, y] in pixel coordinates
    try:
        if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 2:
            x, y = int(evt.index[0]), int(evt.index[1])
        else:
            # Fallback if format is different
            x, y = int(evt.index[0]), int(evt.index[1])
    except (IndexError, TypeError, ValueError) as e:
        print(f"[ERROR] Failed to parse click coordinates: {evt.index}, error: {e}")
        return None, state, f"Error processing click: {str(e)}"
    
    print(f"[CLICK] Detected click at: ({x}, {y})")
    
    # Get the original image (without any markers)
    original_img = state["original_image"]
    if not isinstance(original_img, Image.Image):
        original_img = Image.fromarray(original_img)
    
    # Get image dimensions
    img_width, img_height = original_img.size
    
    # Normalize coordinates (0-1) for storage
    normalized_x = x / img_width
    normalized_y = y / img_height
    
    # Initialize user_clicks if not exists
    if "user_clicks" not in state:
        state["user_clicks"] = []
    
    # Store this click
    click_data = {
        "x": normalized_x,
        "y": normalized_y,
        "pixel_x": x,
        "pixel_y": y
    }
    state["user_clicks"].append(click_data)
    
    # Draw ALL markers on fresh copy of original image
    # This ensures markers don't overlap and we can redraw cleanly
    # Always start from the original image to avoid marker accumulation
    image = original_img.copy()
    draw = ImageDraw.Draw(image)
    
    # Try to load a nice font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw all markers
    for i, click in enumerate(state["user_clicks"]):
        cx = int(click["x"] * img_width)
        cy = int(click["y"] * img_height)
        
        # Draw circle marker with blue outline
        radius = 20
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline="#00BFFF",  # Bright blue
            width=4,
            fill=None  # Transparent fill
        )
        
        # Draw number label with background for visibility
        label = f"#{i+1}"
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font_small)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background rectangle for text
        text_x = cx + radius + 5
        text_y = cy - radius - text_height - 5
        draw.rectangle(
            [text_x - 3, text_y - 3, text_x + text_width + 3, text_y + text_height + 3],
            fill="#00BFFF",
            outline="#00BFFF"
        )
        
        # Draw text
        draw.text((text_x, text_y), label, fill="white", font=font_small)
    
    # Update the display image (but keep original_image unchanged)
    state["display_image"] = image
    
    # Return filepath for Gradio
    filepath = save_image_for_gradio(image)
    
    return filepath, state, f"✓ Marked {len(state['user_clicks'])} area(s). Click more or press 'Compare with AI'!"


def clear_playground(state: Dict):
    """Clear user marks and restore original image"""
    if state is None or state.get("original_image") is None:
        return None, state, "No image loaded."
    
    original = state["original_image"].copy()
    state["display_image"] = original  # Reset display to original
    state["user_clicks"] = []
    
    # Return filepath for Gradio
    filepath = save_image_for_gradio(original)
    
    return filepath, state, "*Marks cleared. Click on the image to mark issues again.*"


def compare_with_ai(image, state: Dict):
    """Compare user clicks with AI detections"""
    if state is None or state.get("original_image") is None:
        return None, state, "Load an image first!", "**Stats:** Attempts: 0 | Avg Score: 0%"
    
    if not state.get("user_clicks"):
        avg_score_text = f"**Stats:** Attempts: {state.get('attempts', 0)} | Avg Score: {state['total_score'] / state['attempts']:.0f}%" if state.get('attempts', 0) > 0 else "**Stats:** Attempts: 0 | Avg Score: 0%"
        return None, state, "Mark some areas first by clicking on the image!", avg_score_text
    
    # Run YOLO on original (unmarked) image
    original = state["original_image"]
    yolo_result = detect_teeth_yolo(original)
    
    # Draw AI boxes on image
    ai_image = None
    ai_boxes = []
    
    if yolo_result.get("success") and yolo_result.get("teeth_found"):
        ai_boxes = yolo_result.get("teeth_found", [])
        ai_image = draw_bounding_boxes(original.copy(), ai_boxes)
    else:
        # No detections - show original image
        ai_image = original.copy()
    
    # Calculate score - how many user clicks fall inside AI boxes?
    hits = 0
    total_ai_detections = len(ai_boxes)
    
    for click in state["user_clicks"]:
        click_x = click["x"]  # Normalized 0-1
        click_y = click["y"]   # Normalized 0-1
        
        # Check if click is inside any AI box
        for box_data in ai_boxes:
            bbox = box_data.get("bbox", [])
            if len(bbox) == 4:
                # bbox is [x_min, y_min, x_max, y_max] in normalized coordinates (0-1)
                x_min, y_min, x_max, y_max = bbox
                
                if (x_min <= click_x <= x_max and y_min <= click_y <= y_max):
                    hits += 1
                    break
    
    # Calculate precision and recall
    user_clicks_count = len(state["user_clicks"])
    
    if user_clicks_count > 0 and total_ai_detections > 0:
        precision = hits / user_clicks_count  # How many of your clicks were correct
        recall = hits / total_ai_detections    # How many AI detections did you find
        score = (precision + recall) / 2 * 100  # F1-like score
    elif total_ai_detections == 0 and user_clicks_count == 0:
        # No issues to find - perfect score
        score = 100
    elif total_ai_detections == 0 and user_clicks_count > 0:
        # You marked issues but AI found none - low score
        score = 20
    elif total_ai_detections > 0 and user_clicks_count == 0:
        # AI found issues but you didn't mark any - low score
        score = 10
    else:
        score = 0
    
    # Update stats
    state["attempts"] = state.get("attempts", 0) + 1
    state["total_score"] = state.get("total_score", 0) + score
    avg_score = state["total_score"] / state["attempts"]
    
    # Generate feedback
    if score >= 80:
        feedback_emoji = "🎉"
        feedback_text = "Excellent!"
    elif score >= 60:
        feedback_emoji = "👍"
        feedback_text = "Good job!"
    elif score >= 40:
        feedback_emoji = "🤔"
        feedback_text = "Getting there!"
    else:
        feedback_emoji = "❌"
        feedback_text = "Keep practicing!"
    
    result_text = f"""
    ### {feedback_emoji} Results: {feedback_text}
    
    **Your Score:** {score:.0f}%
    
    - Your marks: {user_clicks_count}
    - AI detections: {total_ai_detections}
    - Correct hits: {hits}
    
    **AI found:** {yolo_result.get('summary', 'No issues detected')}
    """
    
    stats_text = f"**Stats:** Attempts: {state['attempts']} | Avg Score: {avg_score:.0f}%"
    
    # Return filepath for Gradio
    ai_filepath = save_image_for_gradio(ai_image)
    
    return ai_filepath, state, result_text, stats_text


def create_playground_tab(dataset_manager: TeethDatasetManager):
    """
    Create the Annotation Playground tab
    
    Args:
        dataset_manager: TeethDatasetManager instance for loading images
    
    Returns:
        gr.Tab object with playground UI
    """
    with gr.Tab("🎯 Annotation Playground") as playground_tab:
        gr.Markdown("""
        ### Test Your Diagnostic Skills!
        **Instructions:** 
        1. **Upload an X-ray image** using the uploader below (recommended)
        2. OR click **🎲 Random X-Ray** to try loading from dataset (may be unavailable)
        3. Click on areas where you see dental issues (wisdom teeth, cavities, etc.)
        4. Click **✅ Compare with AI** to see how your marks compare with AI detection
        5. Get a score and feedback!
        
        💡 **Tip:** The more accurate your clicks, the higher your score!
        
        ⚠️ **Note:** The HuggingFace dataset appears to be corrupted. Please upload your own X-ray images for best results.
        """)
        
        # Image upload component (can still use pil for upload, we'll convert)
        with gr.Row():
            image_upload = gr.Image(
                label="📤 Upload X-Ray Image",
                type="filepath",  # Changed to filepath for consistency
                height=200,
                show_label=True
            )
            upload_btn = gr.Button("📥 Load Uploaded Image", variant="primary", size="lg")

        # State management for playground
        # Separate original_image (never modified) from display_image (with markers)
        playground_state = gr.State({
            "original_image": None,  # Store unmarked original (never modified)
            "display_image": None,   # Display image with markers
            "current_index": None,
            "user_clicks": [],
            "attempts": 0,
            "total_score": 0
        })

        with gr.Row():
            random_btn = gr.Button("🎲 Random X-Ray (Dataset)", variant="secondary", size="lg")
            clear_btn = gr.Button("🗑️ Clear Marks", variant="secondary", size="lg")
            compare_btn = gr.Button("✅ Compare with AI", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                # Use gr.Image with .select() event for click detection (Gradio 6.x)
                # Add CSS and JavaScript to prevent image dragging in Firefox
                gr.HTML("""
                <style>
                /* Prevent image dragging in Firefox and other browsers */
                #playground_image img,
                [id*="playground_image"] img,
                .gradio-image img,
                [id*="playground_image"] > div img {
                    user-select: none !important;
                    -webkit-user-select: none !important;
                    -moz-user-select: none !important;
                    -ms-user-select: none !important;
                    pointer-events: auto !important;
                    -webkit-user-drag: none !important;
                    -khtml-user-drag: none !important;
                    -moz-user-drag: none !important;
                    -o-user-drag: none !important;
                    user-drag: none !important;
                    cursor: crosshair !important;
                    -webkit-touch-callout: none !important;
                }
                
                /* Prevent dragging on the container as well */
                #playground_image,
                [id*="playground_image"] {
                    -webkit-user-drag: none !important;
                    -moz-user-drag: none !important;
                    user-drag: none !important;
                }
                </style>
                
                <script>
                // Prevent image dragging in Firefox
                (function() {
                    function preventImageDrag() {
                        // Find all images in the playground
                        const images = document.querySelectorAll('#playground_image img, [id*="playground_image"] img');
                        images.forEach(img => {
                            // Remove draggable attribute
                            img.draggable = false;
                            
                            // Prevent dragstart event
                            img.addEventListener('dragstart', function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                return false;
                            }, false);
                            
                            // Prevent drag event
                            img.addEventListener('drag', function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                return false;
                            }, false);
                        });
                    }
                    
                    // Run on page load
                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', preventImageDrag);
                    } else {
                        preventImageDrag();
                    }
                    
                    // Re-run when image updates
                    const observer = new MutationObserver(function(mutations) {
                        preventImageDrag();
                    });
                    
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true
                    });
                })();
                </script>
                """)
                
                # Use gr.Image with filepath type to fix corruption issues
                playground_image = gr.Image(
                    label="Click on dental issues you spot",
                    type="filepath",  # Changed from "pil" to fix corruption
                    height=500,
                    show_label=True,
                    elem_id="playground_image",
                    interactive=False  # KEY: Disable drag/drop - prevents drag behavior
                )
            
            with gr.Column(scale=1):
                # Show AI result after comparison
                ai_result_image = gr.Image(
                    label="AI Detection Result",
                    type="filepath",  # Changed from "pil" to fix corruption
                    interactive=False,
                    height=500
                )
        
        # Results display
        results_md = gr.Markdown("*Click 'Random X-Ray' to start*")
        
        # Stats
        stats_md = gr.Markdown("**Stats:** Attempts: 0 | Avg Score: 0%")

        # Wire up events
        def load_upload(img, state):
            return load_uploaded_image(img, state)
        
        def load_random(state):
            return load_random_playground_image(state, dataset_manager)
        
        # Upload button - primary method
        upload_btn.click(
            fn=load_upload,
            inputs=[image_upload, playground_state],
            outputs=[playground_image, ai_result_image, playground_state, results_md, stats_md]
        )
        
        # Random button - fallback (may not work if dataset is corrupted)
        random_btn.click(
            fn=load_random,
            inputs=[playground_state],
            outputs=[playground_image, ai_result_image, playground_state, results_md, stats_md]
        )

        clear_btn.click(
            fn=clear_playground,
            inputs=[playground_state],
            outputs=[playground_image, playground_state, results_md]
        )

        compare_btn.click(
            fn=compare_with_ai,
            inputs=[playground_image, playground_state],
            outputs=[ai_result_image, playground_state, results_md, stats_md]
        )

        # Capture click events using Gradio's built-in .select() event
        # This is the proper way to handle clicks on images in Gradio 6.x
        playground_image.select(
            fn=handle_image_click,
            inputs=[playground_state],
            outputs=[playground_image, playground_state, results_md]
        )
    
    return playground_tab

