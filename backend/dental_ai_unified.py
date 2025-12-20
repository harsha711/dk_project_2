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
    multimodal_chat_async
)
from image_utils import (
    parse_vision_response,
    draw_bounding_boxes
)
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
    image: Optional[Image.Image],
    history: List,
    conversation_state: List,
    stored_annotated_images: List  # NEW: Store annotated images persistently
) -> Tuple[List, str, Optional[Image.Image], List, dict, List, List]:
    """
    Process user message and return updated chat history

    Phase 3: Now maintains full conversation state for context-aware responses
    FIXED: Annotated images are preserved across follow-up questions

    Args:
        message: User's text input
        image: Optional uploaded image
        history: Display history (for Gradio chatbot UI)
        conversation_state: Internal conversation state with full context
        stored_annotated_images: Persistent storage for annotated images

    Returns:
        (updated_history, cleared_message, cleared_image, annotated_images_list, gallery_update, updated_conversation_state, updated_stored_images)
    """
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

        # Call models (async) - always pass the most recent image
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        responses = loop.run_until_complete(
            multimodal_chat_async(
                message=message,
                image=current_image,
                conversation_context=context,
                models=models,
                openai_client=openai_client,
                groq_client=groq_client
            )
        )

        loop.close()

        # Handle vision responses differently (with image annotations)
        if mode in ["vision", "vision-followup"]:
            # Parse vision responses and draw bounding boxes
            annotated_images = {}
            annotated_list = []

            # Process vision model responses for annotations
            for model_name, response_text in responses.items():
                if model_name in ["gemini-vision", "groq-vision"]:
                    print(f"\n[VISION DEBUG] {model_name} response preview:")
                    print(f"  {response_text[:200]}...")
                    
                    # Check if Groq Vision is not available
                    if model_name == "groq-vision" and ("not currently available" in response_text.lower() or 
                                                         "not available" in response_text.lower() or
                                                         "not supported" in response_text.lower()):
                        print(f"  ⚠️ Groq Vision is not available")
                        continue

                    # Check if this is a follow-up question
                    if mode == "vision-followup":
                        print(f"  ✅ Follow-up response (natural language): {response_text[:100]}...")
                        # Don't try to parse as JSON for follow-ups
                    else:
                        # Initial analysis - try to parse JSON and draw bounding boxes
                        parsed = parse_vision_response(response_text)

                        if parsed.get('error'):
                            print(f"  ❌ Parse error: {parsed.get('error')}")
                        else:
                            print(f"  ✅ Parsed successfully, found {len(parsed.get('teeth_found', []))} teeth")

                        # If teeth were detected with bounding boxes, draw them
                        if parsed.get('teeth_found') and not parsed.get('error') and current_image:
                            teeth = parsed.get('teeth_found', [])
                            if teeth and isinstance(teeth, list) and len(teeth) > 0:
                                print(f"  🎨 Drawing {len(teeth)} bounding boxes")
                                annotated = draw_bounding_boxes(current_image, teeth)
                                annotated_images[model_name] = annotated
                                if "groq" in model_name:
                                    model_label = "Llama 3.2 Vision"
                                else:
                                    model_label = "Gemini Vision"
                                annotated_list.append((annotated, model_label))

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

.model-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    margin-right: 5px;
}

.vision-result {
    border: 2px solid #4ECDC4;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
"""

# Build the Gradio app
with gr.Blocks(css=custom_css, title="Dental AI Platform") as demo:
    # Header
    gr.HTML("""
    <div class="header-gradient">
        <h1 style="margin: 0; font-size: 2.2rem;">🦷 Dental AI Platform</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Upload dental X-rays • Get wisdom tooth analysis • Ask follow-up questions
        </p>
    </div>
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
                label="📊 Annotated X-Rays (Wisdom Teeth Detected)",
                show_label=True,
                columns=2,
                height=300,
                visible=False
            )

            # Input area
            with gr.Row():
                with gr.Column(scale=1):
                    image_upload = gr.Image(
                        type="pil",
                        label="📎 Upload X-Ray",
                        height=100
                    )

                with gr.Column(scale=9):
                    msg_input = gr.Textbox(
                        placeholder="Ask about wisdom teeth or upload an X-ray...",
                        label="Your Message",
                        lines=2
                    )

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
                        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

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
            Browse, search, and batch-process dental X-rays from Hugging Face.
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
                        prev_btn = gr.Button("⬅️ Previous")
                        next_btn = gr.Button("Next ➡️")

                    random_btn = gr.Button("🎲 Random Sample", variant="secondary")
                    jump_index = gr.Number(label="Jump to Index", value=0, minimum=0, maximum=1205)
                    jump_btn = gr.Button("Go to Index")

            # Navigation events
            next_btn.click(fn=next_sample, inputs=[current_index], outputs=[sample_image, sample_info, current_index])
            prev_btn.click(fn=prev_sample, inputs=[current_index], outputs=[sample_image, sample_info, current_index])
            random_btn.click(fn=random_sample, outputs=[sample_image, sample_info, current_index])
            jump_btn.click(fn=browse_sample, inputs=[jump_index], outputs=[sample_image, sample_info, current_index])

    # Footer
    gr.Markdown("""
    ---
    **Dental AI Platform v2.1** | Unified Chatbot | Powered by Groq, Google Gemini, and Hugging Face
    
    ⚠️ **Note:** Gemini may be rate-limited on free tier. Groq Llama Vision is recommended for consistent results.
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
    print("="*60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )