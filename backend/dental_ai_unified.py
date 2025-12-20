"""
Dental AI Platform - Unified Chatbot Interface
Combines vision and chat capabilities in single conversational interface
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
    conversation_state: List
) -> Tuple[List, str, Optional[Image.Image], List, dict, List]:
    """
    Process user message and return updated chat history

    Phase 3: Now maintains full conversation state for context-aware responses

    Args:
        message: User's text input
        image: Optional uploaded image
        history: Display history (for Gradio chatbot UI)
        conversation_state: Internal conversation state with full context

    Returns:
        (updated_history, cleared_message, cleared_image, annotated_images_list, gallery_update, updated_conversation_state)
    """
    # Handle empty message with no image
    if (not message or not message.strip()) and not image:
        return history, "", None, [], gr.update(visible=False), conversation_state

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
            for entry in reversed(conversation_state):
                if entry.get("role") == "user" and entry.get("image"):
                    current_image = entry["image"]
                    break

        # Build conversation context from state
        context = build_conversation_context(conversation_state, max_turns=5)

        # Call models (async) - always pass the most recent image
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        responses = loop.run_until_complete(
            multimodal_chat_async(
                message=message,
                image=current_image,  # Always use most recent image
                conversation_context=context,
                models=models,
                openai_client=openai_client,
                groq_client=groq_client
            )
        )

        loop.close()

        # Handle vision responses differently (with image annotations)
        # For vision-followup: Gemini Vision analyzes, then GPT-4o and Groq use that analysis
        if mode in ["vision", "vision-followup"]:
            # Parse vision responses and draw bounding boxes
            annotated_images = {}
            annotated_list = []

            # Process vision model responses for annotations
            for model_name, response_text in responses.items():
                if model_name in ["gemini-vision", "groq-vision"]:
                    # Debug logging
                    print(f"\n[VISION DEBUG] {model_name} response preview:")
                    print(f"  {response_text[:200]}...")
                    
                    # Check if Groq Vision is not available
                    if model_name == "groq-vision" and ("not currently available" in response_text.lower() or 
                                                         "not available" in response_text.lower() or
                                                         "not supported" in response_text.lower()):
                        print(f"  ⚠️ Groq Vision is not available - Groq may not support vision models yet")
                        # Continue to show the message in the UI, but don't try to parse/annotate
                        continue

                    # Try to parse structured response
                    parsed = parse_vision_response(response_text)

                    # Debug logging
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
                            # Add to list for inline display with label
                            if "groq" in model_name:
                                model_label = "Llama 3.2 Vision"
                            else:
                                model_label = "Gemini Vision"
                            annotated_list.append((annotated, model_label))
                
                # For vision-followup mode, GPT-4o and Groq responses are handled in formatting
                # They don't need image annotation, just text responses using Gemini's analysis

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

            # Update display history - include images inline in chat
            # Format user message with image if uploaded (dictionary format for Gradio)
            user_message_content = message
            if image:
                # User uploaded image - include it in chat using dictionary format
                # Resize image for chat display to prevent oversized images
                from image_utils import resize_image_for_chat
                resized_user_image = resize_image_for_chat(image, max_width=500, max_height=400)
                history.append({"role": "user", "content": user_message_content, "files": [resized_user_image]})
            else:
                history.append({"role": "user", "content": user_message_content})
            
            # Assistant response with annotated images inline
            assistant_content = formatted_response
            # Include annotated images in the assistant message
            if annotated_list and len(annotated_list) > 0:
                # Show annotated images with bounding boxes in chat
                # For Gradio Chatbot, show the first annotated image inline (with bounding boxes)
                # All annotated images are also available in the gallery below
                first_image = annotated_list[0][0]  # Get first annotated image (with bounding boxes)
                # Resize image for chat display (max 500px width, 400px height)
                from image_utils import resize_image_for_chat
                resized_image = resize_image_for_chat(first_image, max_width=500, max_height=400)
                history.append({"role": "assistant", "content": assistant_content, "files": [resized_image]})
                # Return gallery with all annotated images for full view
                print(f"  ✅ Displaying {len(annotated_list)} annotated image(s) with bounding boxes")
                return history, "", None, annotated_list, gr.update(visible=True), conversation_state
            else:
                # No teeth detected or parsing failed - still show original image for reference
                if current_image:
                    # Resize image for chat display
                    from image_utils import resize_image_for_chat
                    resized_image = resize_image_for_chat(current_image, max_width=500, max_height=400)
                    history.append({"role": "assistant", "content": assistant_content, "files": [resized_image]})
                    print(f"  ⚠️ No wisdom teeth detected - showing original image")
                else:
                    history.append({"role": "assistant", "content": assistant_content})
                return history, "", None, [], gr.update(visible=False), conversation_state
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

            # Update display history - use dictionary format for Gradio Chatbot
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": formatted_response})
            return history, "", None, [], gr.update(visible=False), conversation_state

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        # Log full error details to console
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

        # User-friendly error message
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
        return history, "", None, [], gr.update(visible=False), conversation_state


def clear_conversation():
    """Clear conversation history and state"""
    return [], "", []  # Clear display history, message input, and conversation state


# ============ GRADIO UI ============

custom_css = """
#component-0 {
    max-width: 1400px;
    margin: auto;
    padding-top: 1rem;
}

.header-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}

.chat-container {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 10px;
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

.model-response-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 15px;
    margin: 10px 0;
}

.model-card {
    border: 2px solid;
    border-radius: 8px;
    padding: 12px;
}

.model-card h4 {
    margin-top: 0;
}
"""

with gr.Blocks(title="Dental AI Platform") as demo:

    # Header
    gr.HTML("""
        <div class="header-gradient">
            <h1 style="margin: 0; font-size: 2.8em;">🦷 Dental AI Platform</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                Unified Multi-Model Chatbot for Wisdom Teeth Analysis
            </p>
        </div>
    """)

    with gr.Tabs():

        # ============ TAB 1: UNIFIED CHATBOT ============
        with gr.Tab("🤖 Dental AI Assistant"):
            gr.Markdown("""
            ### Ask questions or upload X-rays - all models respond together

            **What you can do:**
            - 💬 Ask text questions about wisdom teeth
            - 📸 Upload dental X-rays for analysis
            - 🔄 Have follow-up conversations with context ✨ **NEW: Phase 3**

            **Available Models:**
            - 🟢 GPT-4o (OpenAI) - Most capable reasoning (text only)
            - 🔵 Gemini (Google) - Fast and efficient
            - 🔵 Gemini Vision - X-ray analysis and vision tasks
            - 🟠 Groq Llama3 - Ultra-fast inference (text only)
            - 🟠 Llama 3.2 Vision (Groq) - Vision analysis (may not be available)
            """)

            # Conversation state (hidden from user, tracks full context)
            conversation_state = gr.State([])

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

            # Event handlers
            send_btn.click(
                fn=process_chat_message,
                inputs=[msg_input, image_upload, chatbot, conversation_state],
                outputs=[chatbot, msg_input, image_upload, annotated_gallery, annotated_gallery, conversation_state]
            )

            msg_input.submit(
                fn=process_chat_message,
                inputs=[msg_input, image_upload, chatbot, conversation_state],
                outputs=[chatbot, msg_input, image_upload, annotated_gallery, annotated_gallery, conversation_state]
            )

            clear_btn.click(
                fn=clear_conversation,
                outputs=[chatbot, msg_input, conversation_state]
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
    **Dental AI Platform v2.0** | Unified Chatbot | Powered by OpenAI, Google, Groq, and Hugging Face
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
    print("     - 3 models respond in parallel")
    print("  ✅ Tab 2: Dataset Explorer (1,206 samples)")
    print("="*60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=custom_css
    )
