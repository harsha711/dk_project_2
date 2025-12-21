"""
Dental AI Platform - Unified Chatbot Interface
Combines vision and chat capabilities in single conversational interface

FIXED: Annotated images are now preserved during follow-up questions
"""
import os
import asyncio
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
from typing import List, Tuple, Optional
import time

# Import utility modules
from api_utils import (
    init_clients,
    multimodal_chat_async,
    detect_teeth_yolo
)
from image_utils import draw_bounding_boxes
from dataset_utils import TeethDatasetManager
from multimodal_utils import (
    route_message,
    build_conversation_context,
    format_multi_model_response,
    SYSTEM_PROMPT
)

# Load environment variables
load_dotenv()

# Initialize API clients globally
openai_client, groq_client = init_clients()

# Initialize dataset manager
dataset_manager = TeethDatasetManager()


# ============ PHASE 3: CONVERSATION HISTORY ============

def process_chat_message(
    message: str,
    image_file,  # Now receives file path from gr.File
    history: List,
    conversation_state: List,
    stored_annotated_images: List  # NEW: Store annotated images persistently
) -> Tuple[List, str, None, List, dict, List, List]:
    """
    Process user message and return updated chat history

    Phase 3: Now maintains full conversation state for context-aware responses
    FIXED: Annotated images are preserved across follow-up questions

    Args:
        message: User's text input
        image_file: File path from gr.File upload
        history: Display history (for Gradio chatbot UI)
        conversation_state: Internal conversation state with full context
        stored_annotated_images: Persistent storage for annotated images

    Returns:
        (updated_history, cleared_message, cleared_image, annotated_images_list, gallery_update, updated_conversation_state, updated_stored_images)
    """
    # Convert file to PIL Image if provided
    image = None
    if image_file is not None:
        try:
            image = Image.open(image_file)
        except Exception as e:
            print(f"Error loading image: {e}")
            image = None
    
    # Handle empty message with no image
    if (not message or not message.strip()) and not image:
        # Return stored images to keep gallery visible
        gallery_visible = len(stored_annotated_images) > 0
        return history, "", None, stored_annotated_images, gr.update(visible=gallery_visible), conversation_state, stored_annotated_images

    # If no message but image provided, use default prompt
    if not message or not message.strip():
        message = "Analyze this dental X-ray for wisdom teeth."

    try:
        # Add user message to conversation state
        user_entry = {
            "role": "user",
            "content": message,
            "image": image,
            "timestamp": time.time()
        }
        conversation_state.append(user_entry)

        # Determine which models to use (now checks conversation_state for recent images)
        mode, models = route_message(message, image, conversation_state)

        # Always find the most recent X-ray image to include in API calls
        current_image = image
        if not current_image:
            # Find most recent image in conversation history
            print(f"[IMAGE RETRIEVAL] No image in current message, searching conversation history...")
            print(f"[IMAGE RETRIEVAL] Conversation state has {len(conversation_state)} entries")
            
            for i in range(len(conversation_state) - 2, -1, -1):
                entry = conversation_state[i]
                if entry.get("role") == "user" and entry.get("image"):
                    current_image = entry["image"]
                    print(f"[IMAGE RETRIEVAL] ✅ Found image from entry {i} (size: {current_image.size if current_image else 'None'})")
                    break
        
        if current_image:
            print(f"[IMAGE RETRIEVAL] ✅ Using image for vision analysis (size: {current_image.size})")
        else:
            print(f"[IMAGE RETRIEVAL] ⚠️ WARNING: No image found for vision analysis!")

        # Build conversation context from state
        context = build_conversation_context(conversation_state, max_turns=5)

        # YOLO DETECTION FIRST (for initial vision analysis only)
        yolo_detections = None
        yolo_summary = None
        if mode == "vision" and current_image:
            print("\n[YOLO DETECTION] Running YOLOv8 for accurate bounding box detection...")
            # Using default thresholds with class-specific filtering and spatial validation
            yolo_result = detect_teeth_yolo(current_image)

            if yolo_result.get("success") and yolo_result.get("teeth_found"):
                yolo_detections = yolo_result.get("teeth_found", [])
                print(f"  ✅ YOLO detected {len(yolo_detections)} teeth/features")

                # Create detection summary for text models
                detection_details = []
                for det in yolo_detections:
                    class_name = det.get('class_name', 'unknown')
                    position = det.get('position', 'unknown')
                    confidence = det.get('confidence', 0)
                    detection_details.append(f"{class_name} at {position} ({confidence:.0%} confidence)")

                yolo_summary = f"""YOLO Object Detection Results:
- Total detections: {len(yolo_detections)}
- Findings: {', '.join(detection_details)}

Based on these YOLO detections, please provide a brief clinical analysis (2-3 sentences) about:
1. What these findings indicate about the dental condition
2. Any concerns or recommendations
3. Suggested follow-up actions"""

                print(f"  📝 YOLO summary created for text model analysis")
            else:
                print(f"  ⚠️ YOLO detection failed or found no teeth: {yolo_result.get('summary', 'Unknown error')}")
                yolo_summary = f"YOLO Detection: {yolo_result.get('summary', 'No teeth detected')}"

        # Modify context to include YOLO results for text models
        analysis_context = context.copy()
        if yolo_summary and mode == "vision":
            # Add YOLO results to the last user message for text model analysis
            if analysis_context and analysis_context[-1]['role'] == 'user':
                original_content = analysis_context[-1]['content']
                analysis_context[-1] = {
                    "role": "user",
                    "content": f"{original_content}\n\n{yolo_summary}"
                }
                print(f"[CONTEXT DEBUG] Added YOLO summary to user message")
                print(f"  Original: {original_content[:100]}...")
                print(f"  With YOLO: {analysis_context[-1]['content'][:200]}...")
            else:
                print(f"[CONTEXT DEBUG] ⚠️ Could not add YOLO summary - context structure issue")

        # Call models (async) - now with YOLO context for vision mode
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        responses = loop.run_until_complete(
            multimodal_chat_async(
                message=message,
                image=current_image,
                conversation_context=analysis_context,
                models=models,
                openai_client=openai_client,
                groq_client=groq_client
            )
        )

        loop.close()

        # Handle vision mode (initial image upload with YOLO detection)
        if mode == "vision":
            # Parse vision responses and draw bounding boxes
            annotated_images = {}
            annotated_list = []

            # Create annotated image from YOLO detections
            if yolo_detections:
                print(f"  🎨 Creating annotated image with {len(yolo_detections)} YOLO detections")
                yolo_annotated = draw_bounding_boxes(current_image, yolo_detections, show_confidence=True)
                annotated_images["yolo"] = yolo_annotated
                annotated_list.append((yolo_annotated, "YOLOv8 Detection"))
                print(f"  ✅ Annotated image created")

            # Format vision response with annotated images
            from multimodal_utils import format_vision_response
            formatted_response = format_vision_response(responses, annotated_images)

            # Add assistant response to conversation state
            assistant_entry = {
                "role": "assistant",
                "model_responses": responses,
                "timestamp": time.time()
            }
            conversation_state.append(assistant_entry)

            # Update display history
            from image_utils import resize_image_for_chat
            image_to_display = image if image else current_image
            
            print(f"[DISPLAY] Vision mode - image_to_display: {image_to_display is not None}")
            
            if image_to_display:
                resized_user_image = resize_image_for_chat(image_to_display, max_width=500, max_height=400)
                print(f"[DISPLAY] ✅ Adding image to user message (size: {resized_user_image.size})")
                history.append({"role": "user", "content": message, "files": [resized_user_image]})
            else:
                print(f"[DISPLAY] ⚠️ No image to display in vision mode")
                history.append({"role": "user", "content": message})
            
            # Assistant response with annotated images inline
            assistant_content = formatted_response
            
            # FIXED: Update stored images if new annotations were created
            if annotated_list and len(annotated_list) > 0:
                # NEW annotations - update stored images
                stored_annotated_images = annotated_list
                first_image = annotated_list[0][0]
                resized_image = resize_image_for_chat(first_image, max_width=500, max_height=400)
                history.append({"role": "assistant", "content": assistant_content, "files": [resized_image]})
                print(f"  ✅ Displaying {len(annotated_list)} NEW annotated image(s)")
                return history, "", None, annotated_list, gr.update(visible=True), conversation_state, stored_annotated_images
            else:
                # No new annotations - KEEP existing stored images visible
                if current_image:
                    resized_image = resize_image_for_chat(current_image, max_width=500, max_height=400)
                    history.append({"role": "assistant", "content": assistant_content, "files": [resized_image]})
                    print(f"  ⚠️ No new wisdom teeth detected - keeping existing annotated images")
                else:
                    history.append({"role": "assistant", "content": assistant_content})
                
                # FIXED: Return stored images instead of empty list
                gallery_visible = len(stored_annotated_images) > 0
                return history, "", None, stored_annotated_images, gr.update(visible=gallery_visible), conversation_state, stored_annotated_images
        else:
            # Text-only responses - use standard formatting
            formatted_response = format_multi_model_response(responses)

            # Add assistant response to conversation state
            assistant_entry = {
                "role": "assistant",
                "model_responses": responses,
                "timestamp": time.time()
            }
            conversation_state.append(assistant_entry)

            # Update display history
            from image_utils import resize_image_for_chat
            image_to_display = image if image else current_image
            
            print(f"[DISPLAY] Text-only mode - image_to_display: {image_to_display is not None}")
            
            if image_to_display:
                resized_image = resize_image_for_chat(image_to_display, max_width=500, max_height=400)
                print(f"[DISPLAY] ✅ Adding image to user message (size: {resized_image.size})")
                history.append({"role": "user", "content": message, "files": [resized_image]})
                history.append({"role": "assistant", "content": formatted_response, "files": [resized_image]})
            else:
                print(f"[DISPLAY] ⚠️ No image to display in text-only mode")
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": formatted_response})
            
            # FIXED: Keep stored images visible for text-only follow-ups
            gallery_visible = len(stored_annotated_images) > 0
            return history, "", None, stored_annotated_images, gr.update(visible=gallery_visible), conversation_state, stored_annotated_images

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        print("=" * 80)
        print("❌ ERROR in process_chat_message")
        print("=" * 80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"\nInput Message: {message}")
        print(f"Has Image: {image is not None}")
        print(f"Conversation State Length: {len(conversation_state)}")
        print("\nFull Traceback:")
        print(error_details)
        print("=" * 80)

        error_msg = f"""❌ **Error Occurred**

**Type:** {type(e).__name__}
**Message:** {str(e)}

Please check the console for detailed error logs.
If the error persists, try:
1. Clearing the conversation
2. Refreshing the page
3. Checking your API keys in .env file"""

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        
        # FIXED: Keep stored images on error
        gallery_visible = len(stored_annotated_images) > 0
        return history, "", None, stored_annotated_images, gr.update(visible=gallery_visible), conversation_state, stored_annotated_images


def clear_conversation():
    """Clear conversation history and state"""
    return [], "", [], [], gr.update(visible=False), []  # Also clear stored images


# ============ GRADIO UI ============

custom_css = """
#component-0 {
    max-width: 1400px;
    margin: auto;
    padding-top: 1rem;
}

.header-gradient {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
}

.chat-container {
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}

/* Constrain images in chat to prevent oversized display */
.chat-container img,
.message img,
[class*="message"] img,
[class*="chat"] img {
    max-width: 500px !important;
    max-height: 400px !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 8px;
    margin: 5px 0;
}

/* Ensure chat messages don't overflow */
.message {
    max-width: 100% !important;
    overflow: hidden !important;
}

/* Smaller gallery images with proper sizing */
.gradio-gallery {
    max-height: 450px !important;
}

.gradio-gallery img {
    max-width: 600px !important;
    max-height: 400px !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    cursor: pointer !important;
    transition: transform 0.2s ease;
}

.gradio-gallery img:hover {
    transform: scale(1.02);
}

/* Gallery container styling */
.gradio-gallery .grid-wrap {
    max-height: 450px !important;
    overflow-y: auto !important;
}

/* Modal overlay for fullscreen image */
.image-modal-overlay {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.95);
    z-index: 9999;
    justify-content: center;
    align-items: center;
    cursor: zoom-out;
}

.image-modal-overlay.active {
    display: flex !important;
}

.image-modal-content {
    max-width: 95vw;
    max-height: 95vh;
    object-fit: contain;
    border-radius: 8px;
}

.modal-close-btn {
    position: absolute;
    top: 20px;
    right: 30px;
    font-size: 40px;
    color: white;
    cursor: pointer;
    background: none;
    border: none;
    z-index: 10000;
}

.modal-close-btn:hover {
    color: #ff6b6b;
}
"""

# Build the Gradio app with dark theme
with gr.Blocks(css=custom_css, title="Dental AI Platform", theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue").set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
)) as demo:
    # Header
    gr.HTML("""
    <div class="header-gradient">
        <h1 style="margin: 0; font-size: 2.2rem;">🦷 Dental AI Platform</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Upload dental X-rays • Get wisdom tooth analysis • Ask follow-up questions
        </p>
    </div>

    <!-- Modal for fullscreen image viewing -->
    <div id="imageModal" class="image-modal-overlay" onclick="closeModal()">
        <button class="modal-close-btn" onclick="closeModal()">&times;</button>
        <img id="modalImage" class="image-modal-content" src="" alt="Fullscreen view">
    </div>

    <script>
        // Function to open modal with fullscreen image
        function openModal(imgSrc) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.classList.add('active');
            modalImg.src = imgSrc;
            document.body.style.overflow = 'hidden'; // Prevent scrolling
        }

        // Function to close modal
        function closeModal() {
            const modal = document.getElementById('imageModal');
            modal.classList.remove('active');
            document.body.style.overflow = 'auto'; // Re-enable scrolling
        }

        // Add click handlers to gallery images after page loads
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                addImageClickHandlers();
            }, 1000);
        });

        // Add click handlers to all gallery images
        function addImageClickHandlers() {
            const galleryImages = document.querySelectorAll('.gradio-gallery img');
            galleryImages.forEach(img => {
                if (!img.hasAttribute('data-modal-listener')) {
                    img.setAttribute('data-modal-listener', 'true');
                    img.addEventListener('click', function(e) {
                        e.stopPropagation();
                        openModal(this.src);
                    });
                }
            });
        }

        // Re-run handler attachment when gallery updates (mutation observer)
        const observer = new MutationObserver(function(mutations) {
            addImageClickHandlers();
        });

        // Start observing the document for changes
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Close modal on ESC key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
    </script>
    """)

    with gr.Tabs():
        # ============ TAB 1: UNIFIED CHATBOT ============
        with gr.Tab("💬 Wisdom Tooth Assistant"):
            gr.Markdown("""
            ### Chat with our AI dental assistants about wisdom teeth
            **Upload an X-ray** for analysis or **ask questions** about wisdom teeth.
            Multiple AI models will respond simultaneously for comparison.
            
            📌 **Note:** After uploading an X-ray, you can ask follow-up questions and the annotated image will stay visible!
            """)

            # Conversation state (hidden from user, tracks full context)
            conversation_state = gr.State([])
            
            # NEW: Store annotated images persistently
            stored_annotated_images = gr.State([])

            # Chat display
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                elem_classes=["chat-container"]
            )

            # Annotated images gallery (for vision results)
            annotated_gallery = gr.Gallery(
                label="📊 Annotated X-Rays (Click to maximize)",
                show_label=True,
                columns=1,
                rows=1,
                height=350,
                object_fit="contain",
                visible=False,
                allow_preview=True,
                preview=True
            )

            # Input area
            with gr.Row():
                with gr.Column(scale=8):
                    msg_input = gr.Textbox(
                        placeholder="Ask about wisdom teeth or upload an X-ray image...",
                        label="Your Message",
                        lines=2,
                        show_label=False
                    )

                    with gr.Row():
                        image_upload = gr.File(
                            label="📎 X-Ray",
                            file_types=["image"],
                            file_count="single",
                            scale=1
                        )
                        clear_btn = gr.Button("🗑️ Clear", scale=1, size="lg")
                        send_btn = gr.Button("Send ➤", variant="primary", scale=2, size="lg")

            # Example questions
            gr.Examples(
                examples=[
                    ["What are the common symptoms of impacted wisdom teeth?"],
                    ["When should wisdom teeth be extracted?"],
                    ["What is the recovery time after wisdom tooth removal?"],
                    ["How do I know if my wisdom teeth are coming in?"],
                    ["What are the risks of leaving impacted wisdom teeth untreated?"]
                ],
                inputs=msg_input,
                label="💡 Example Questions"
            )

            # Event handlers - UPDATED to include stored_annotated_images
            send_btn.click(
                fn=process_chat_message,
                inputs=[msg_input, image_upload, chatbot, conversation_state, stored_annotated_images],
                outputs=[chatbot, msg_input, image_upload, annotated_gallery, annotated_gallery, conversation_state, stored_annotated_images]
            )

            msg_input.submit(
                fn=process_chat_message,
                inputs=[msg_input, image_upload, chatbot, conversation_state, stored_annotated_images],
                outputs=[chatbot, msg_input, image_upload, annotated_gallery, annotated_gallery, conversation_state, stored_annotated_images]
            )

            clear_btn.click(
                fn=clear_conversation,
                outputs=[chatbot, msg_input, conversation_state, stored_annotated_images, annotated_gallery, stored_annotated_images]
            )

        # ============ TAB 2: DATASET EXPLORER ============
        with gr.Tab("📊 Dataset Explorer"):
            gr.Markdown("""
            ### Explore the RayanAi Dental X-Ray Dataset (1,206 samples)
            Browse samples and **send them directly to AI for analysis**!
            
            1. Click **Load Dataset** to fetch from Hugging Face
            2. Browse using Previous/Next or Random
            3. Click **🔬 Analyze with AI** to get wisdom tooth detection
            """)

            with gr.Row():
                load_dataset_btn = gr.Button("📥 Load Dataset from Hugging Face", variant="primary", size="lg")

            dataset_info = gr.Markdown(value="Click 'Load Dataset' to start...")

            def load_hf_dataset():
                result = dataset_manager.load_dataset()
                if result["success"]:
                    stats = dataset_manager.get_dataset_stats()
                    if stats["success"]:
                        return f"""✅ **Dataset Loaded Successfully**

**Total Samples:** {stats['total_samples']}
**Label 0 (images):** {stats['label_0_count']}
**Label 1 (labels):** {stats['label_1_count']}
**Image Size:** {stats['image_size']}
**Dataset:** {stats['dataset_name']}

Use the navigation buttons to explore samples!"""
                    else:
                        return result["message"]
                else:
                    return result["message"]

            def browse_sample(index: int):
                result = dataset_manager.get_sample(index)
                if result["success"]:
                    info = f"""**Sample #{result['index']} of {result['total']}**

**Label:** {result['label']}
**Class:** {"images" if result['label'] == 0 else "labels"}"""
                    return result['image'], info, result['index']
                else:
                    return None, f"❌ {result['error']}", index

            def next_sample(current_idx: int):
                if dataset_manager.dataset is None:
                    return None, "Load dataset first", 0
                total = len(dataset_manager.dataset['train'])
                next_idx = min(current_idx + 1, total - 1)
                return browse_sample(next_idx)

            def prev_sample(current_idx: int):
                if dataset_manager.dataset is None:
                    return None, "Load dataset first", 0
                prev_idx = max(current_idx - 1, 0)
                return browse_sample(prev_idx)

            def random_sample():
                result = dataset_manager.get_random_sample()
                if result["success"]:
                    info = f"""**Random Sample #{result['index']} of {result['total']}**

**Label:** {result['label']}
**Class:** {"images" if result['label'] == 0 else "labels"}"""
                    return result['image'], info, result['index']
                else:
                    return None, f"❌ {result['error']}", 0

            load_dataset_btn.click(fn=load_hf_dataset, outputs=[dataset_info])

            gr.Markdown("---\n### 🔍 Browse Samples")

            with gr.Row():
                with gr.Column(scale=2):
                    sample_image = gr.Image(label="Sample Image", height=512)

                with gr.Column(scale=1):
                    sample_info = gr.Markdown(value="Load dataset and navigate samples...")
                    current_index = gr.Number(value=0, label="Current Index", visible=False)

                    with gr.Row():
                        prev_btn = gr.Button("⬅️ Previous", scale=1)
                        next_btn = gr.Button("Next ➡️", scale=1)

                    random_btn = gr.Button("🎲 Random Sample", variant="secondary", size="lg")
                    jump_index = gr.Number(label="Jump to Index", value=0, minimum=0, maximum=1205)
                    jump_btn = gr.Button("Go to Index", size="lg")
                    
                    gr.Markdown("---")
                    analyze_btn = gr.Button("🔬 Analyze with AI", variant="primary", size="lg")
                    analyze_status = gr.Markdown(value="")
            
            # Analysis results section - shows results right here in Dataset Explorer
            gr.Markdown("---\n### 🤖 AI Analysis Results")
            analysis_output = gr.Markdown(value="*Click 'Analyze with AI' on a sample to see results here*")
            analysis_gallery = gr.Gallery(
                label="📊 Annotated X-Ray (Click to maximize)",
                show_label=True,
                columns=1,
                rows=1,
                height=350,
                object_fit="contain",
                visible=False,
                allow_preview=True,
                preview=True
            )

            # Function to send dataset image to chatbot for analysis
            def analyze_dataset_image(image, current_idx):
                if image is None:
                    return "❌ No image loaded. Please load the dataset and select a sample first."
                return f"✅ **Sample #{int(current_idx)}** sent to AI for analysis! Check the results below."
            
            def run_analysis_on_sample(image, history, conversation_state, stored_annotated_images):
                """Run AI analysis on the dataset sample"""
                if image is None:
                    return history, stored_annotated_images, gr.update(visible=False), conversation_state, stored_annotated_images, "❌ No image to analyze", "❌ No image to analyze", [], gr.update(visible=False)
                
                # Convert numpy array to PIL Image if needed
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                
                # Save image temporarily for processing
                import tempfile
                import os
                temp_path = os.path.join(tempfile.gettempdir(), "dataset_sample.png")
                image.save(temp_path)
                
                # Call the main processing function
                result = process_chat_message(
                    message="Analyze this dental X-ray from the dataset. Identify any wisdom teeth and their condition.",
                    image_file=temp_path,
                    history=history,
                    conversation_state=conversation_state,
                    stored_annotated_images=stored_annotated_images
                )
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                # Extract the analysis text from the last assistant message in history
                analysis_text = "Analysis complete!"
                if result[0] and len(result[0]) > 0:
                    last_msg = result[0][-1]
                    if isinstance(last_msg, dict) and last_msg.get('role') == 'assistant':
                        analysis_text = last_msg.get('content', 'Analysis complete!')
                
                # Get annotated images - same format as chatbot: list of (image, label) tuples
                annotated_imgs = result[3] if result[3] else []
                gallery_visible = len(annotated_imgs) > 0
                
                # Return: history, stored_images, gallery_update, conversation_state, stored_images, status, analysis_text, analysis_gallery_images, analysis_gallery_visible
                return result[0], result[3], result[4], result[5], result[6], "✅ Analysis complete!", analysis_text, annotated_imgs, gr.update(visible=gallery_visible)

            # Navigation events
            next_btn.click(fn=next_sample, inputs=[current_index], outputs=[sample_image, sample_info, current_index])
            prev_btn.click(fn=prev_sample, inputs=[current_index], outputs=[sample_image, sample_info, current_index])
            random_btn.click(fn=random_sample, outputs=[sample_image, sample_info, current_index])
            jump_btn.click(fn=browse_sample, inputs=[jump_index], outputs=[sample_image, sample_info, current_index])
            
            # Analyze button - sends to chatbot AND shows results here
            analyze_btn.click(
                fn=run_analysis_on_sample,
                inputs=[sample_image, chatbot, conversation_state, stored_annotated_images],
                outputs=[chatbot, annotated_gallery, annotated_gallery, conversation_state, stored_annotated_images, analyze_status, analysis_output, analysis_gallery, analysis_gallery]
            )

    # Footer
    gr.Markdown("""
    ---
    **Dental AI Platform v2.3** | Multi-Model Chatbot + YOLO Detection | Powered by GPT-4o-mini, Llama 3.3, Mixtral 8x7B, and YOLOv8
    """)


# ============ LAUNCH APP ============

if __name__ == "__main__":
    print("🚀 Starting Dental AI Platform (Unified Chatbot)...")
    print("📍 Server: http://localhost:7860")
    print("\n" + "="*60)
    print("FEATURES:")
    print("  ✅ Tab 1: Unified Chatbot (Text + Image)")
    print("     - Ask questions about wisdom teeth")
    print("     - Upload X-rays for analysis")
    print("     - Annotated images persist during follow-ups")
    print("  ✅ Tab 2: Dataset Explorer (1,206 samples)")
    print("     - Browse HuggingFace dental X-ray dataset")
    print("     - One-click AI analysis on any sample")
    print("="*60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )