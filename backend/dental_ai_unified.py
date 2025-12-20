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


# ============ PHASE 1: BASIC TEXT CHAT ============

def process_chat_message(
    message: str,
    image: Optional[Image.Image],
    history: List
) -> Tuple[List, str, Optional[Image.Image], List, dict]:
    """
    Process user message and return updated chat history

    Args:
        message: User's text input
        image: Optional uploaded image
        history: Current chat history

    Returns:
        (updated_history, cleared_message, cleared_image, annotated_images_list, gallery_update)
    """
    # Handle empty message with no image
    if (not message or not message.strip()) and not image:
        return history, "", None, [], gr.update(visible=False)

    # If no message but image provided, use default prompt
    if not message or not message.strip():
        message = "Analyze this dental X-ray for wisdom teeth."

    try:
        # Determine which models to use
        mode, models = route_message(message, image, history)

        # Build conversation context
        context = build_conversation_context(history, max_turns=5)

        # Call models (async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        responses = loop.run_until_complete(
            multimodal_chat_async(
                message=message,
                image=image,
                conversation_context=context,
                models=models,
                openai_client=openai_client,
                groq_client=groq_client
            )
        )

        loop.close()

        # Handle vision responses differently (with image annotations)
        if mode in ["vision", "vision-followup"] and image:
            # Parse vision responses and draw bounding boxes
            annotated_images = {}
            annotated_list = []

            for model_name, response_text in responses.items():
                if model_name in ["gpt4-vision", "gemini-vision"]:
                    # Try to parse structured response
                    parsed = parse_vision_response(response_text)

                    # If teeth were detected with bounding boxes, draw them
                    if parsed.get('teeth_found') and not parsed.get('error'):
                        teeth = parsed.get('teeth_found', [])
                        if teeth and isinstance(teeth, list):
                            annotated = draw_bounding_boxes(image, teeth)
                            annotated_images[model_name] = annotated
                            # Add to list for gallery with label
                            model_label = "GPT-4o Vision" if "gpt4" in model_name else "Gemini Vision"
                            annotated_list.append((annotated, model_label))

            # Format vision response with annotated images
            from multimodal_utils import format_vision_response
            formatted_response = format_vision_response(responses, annotated_images)

            # Store image in history for follow-up questions (Phase 3)
            # For now, just display the response
            history.append((message, formatted_response))

            # Show gallery if we have annotated images
            if annotated_list:
                return history, "", None, annotated_list, gr.update(visible=True)
            else:
                return history, "", None, [], gr.update(visible=False)
        else:
            # Text-only responses - use standard formatting
            formatted_response = format_multi_model_response(responses)
            history.append((message, formatted_response))
            return history, "", None, [], gr.update(visible=False)

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nDetails: {type(e).__name__}"
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in process_chat_message: {error_details}")
        history.append((message, error_msg))
        return history, "", None, [], gr.update(visible=False)


def clear_conversation():
    """Clear conversation history"""
    return [], ""


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
            - 🔄 Have follow-up conversations with context

            **Available Models:**
            - 🟢 GPT-4o (OpenAI) - Most capable reasoning
            - 🔵 Gemini (Google) - Fast and efficient
            - 🟠 Groq Llama3 - Ultra-fast inference
            """)

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
                inputs=[msg_input, image_upload, chatbot],
                outputs=[chatbot, msg_input, image_upload, annotated_gallery, annotated_gallery]
            )

            msg_input.submit(
                fn=process_chat_message,
                inputs=[msg_input, image_upload, chatbot],
                outputs=[chatbot, msg_input, image_upload, annotated_gallery, annotated_gallery]
            )

            clear_btn.click(
                fn=clear_conversation,
                outputs=[chatbot, msg_input]
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
