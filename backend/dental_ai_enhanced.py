"""
Enhanced Dental AI Platform with Dataset Integration
Three tabs: Wisdom Tooth Detection, Multi-Model Chatbot, Dataset Explorer
"""
import os
import asyncio
import gradio as gr
from dotenv import load_dotenv
from PIL import Image

# Import our utility modules
from api_utils import (
    init_clients,
    analyze_xray_gpt4v,
    analyze_xray_gemini,
    chat_all_models
)
from image_utils import (
    parse_vision_response,
    draw_bounding_boxes,
)
from dataset_utils import (
    TeethDatasetManager,
    batch_analyze_samples
)

# Load environment variables
load_dotenv()

# Initialize API clients globally
openai_client, groq_client = init_clients()

# Initialize dataset manager
dataset_manager = TeethDatasetManager()


# ============ TAB 1: WISDOM TOOTH DETECTION ============

def process_xray(image: Image.Image, model_choice: str) -> tuple:
    """Process dental X-ray image with selected vision model"""
    if image is None:
        return None, "Please upload an X-ray image first."

    try:
        # Call selected vision API
        if model_choice == "GPT-4o Vision":
            result = analyze_xray_gpt4v(image, openai_client)
        else:  # Gemini Vision
            result = analyze_xray_gemini(image)

        if not result["success"]:
            return None, f"❌ Error from {result['model']}: {result['error']}"

        # Parse the response
        parsed = parse_vision_response(result["response"])

        if "error" in parsed:
            return image, f"⚠️ Could not parse response:\n\n{parsed['raw_response']}"

        # Extract detections
        teeth_found = parsed.get("teeth_found", [])
        summary = parsed.get("summary", "Analysis complete")

        if not teeth_found:
            return image, f"ℹ️ No wisdom teeth detected in the image.\n\nSummary: {summary}"

        # Draw bounding boxes
        annotated_image = draw_bounding_boxes(image, teeth_found)

        # Create analysis text
        analysis_text = f"**{result['model']} Analysis**\n\n"
        analysis_text += f"**Summary:** {summary}\n\n"
        analysis_text += f"**Wisdom Teeth Found:** {len(teeth_found)}\n\n"

        for i, tooth in enumerate(teeth_found, 1):
            analysis_text += f"**Tooth {i}:**\n"
            analysis_text += f"- Position: {tooth.get('position', 'Unknown')}\n"
            analysis_text += f"- Description: {tooth.get('description', 'N/A')}\n"
            analysis_text += f"- Bounding Box: {tooth.get('bbox', [])}\n\n"

        return annotated_image, analysis_text

    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"


# ============ TAB 2: MULTI-MODEL CHATBOT ============

def chat_with_all_models(query: str) -> tuple:
    """Send query to all 3 models in parallel and return responses"""
    if not query or not query.strip():
        empty_msg = "Please enter a question first."
        return empty_msg, empty_msg, empty_msg

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            chat_all_models(query, openai_client, groq_client)
        )
        loop.close()

        openai_result, gemini_result, groq_result = results

        def format_response(result: dict) -> str:
            if result["success"]:
                return f"✅ **{result['model']}**\n\n{result['response']}"
            else:
                return f"❌ **{result['model']}**\n\n{result['response']}"

        return (
            format_response(openai_result),
            format_response(gemini_result),
            format_response(groq_result)
        )

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg, error_msg


# ============ TAB 3: DATASET EXPLORER ============

def load_hf_dataset() -> str:
    """Load the Hugging Face dataset"""
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


def browse_sample(index: int) -> tuple:
    """Browse dataset by index"""
    result = dataset_manager.get_sample(index)

    if result["success"]:
        info = f"""**Sample #{result['index']} of {result['total']}**

**Label:** {result['label']}
**Class:** {"images" if result['label'] == 0 else "labels"}
"""
        return result['image'], info, result['index']
    else:
        return None, f"❌ {result['error']}", index


def next_sample(current_idx: int) -> tuple:
    """Navigate to next sample"""
    if dataset_manager.dataset is None:
        return None, "Load dataset first", 0

    total = len(dataset_manager.dataset['train'])
    next_idx = min(current_idx + 1, total - 1)
    return browse_sample(next_idx)


def prev_sample(current_idx: int) -> tuple:
    """Navigate to previous sample"""
    if dataset_manager.dataset is None:
        return None, "Load dataset first", 0

    prev_idx = max(current_idx - 1, 0)
    return browse_sample(prev_idx)


def random_sample() -> tuple:
    """Get random sample"""
    result = dataset_manager.get_random_sample()

    if result["success"]:
        info = f"""**Random Sample #{result['index']} of {result['total']}**

**Label:** {result['label']}
**Class:** {"images" if result['label'] == 0 else "labels"}
"""
        return result['image'], info, result['index']
    else:
        return None, f"❌ {result['error']}", 0


def batch_process_samples(start_idx: int, batch_size: int, model_choice: str, progress=gr.Progress()) -> str:
    """Process a batch of samples from the dataset"""
    if dataset_manager.dataset is None:
        return "❌ Please load the dataset first!"

    progress(0, desc="Loading samples...")

    # Get batch
    batch_result = dataset_manager.get_batch(start_idx, batch_size)

    if not batch_result["success"]:
        return f"❌ {batch_result['error']}"

    samples = batch_result["samples"]
    progress(0.3, desc=f"Analyzing {len(samples)} samples...")

    # Batch analyze
    results = batch_analyze_samples(samples, model_choice, openai_client, groq_client)

    progress(0.9, desc="Formatting results...")

    # Format results
    output = f"**Batch Analysis Complete**\n\n"
    output += f"**Samples Processed:** {len(results)}\n"
    output += f"**Model Used:** {model_choice}\n\n"
    output += "---\n\n"

    successful = 0
    for result in results:
        if result["success"]:
            successful += 1
            parsed = result["parsed"]
            teeth_count = len(parsed.get("teeth_found", []))
            output += f"**Sample #{result['index']}**: {teeth_count} wisdom teeth detected\n"
        else:
            output += f"**Sample #{result['index']}**: ❌ {result['error']}\n"

    output += f"\n**Success Rate:** {successful}/{len(results)}"

    progress(1.0, desc="Done!")
    return output


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

.response-box {
    min-height: 300px;
    max-height: 500px;
    overflow-y: auto;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    background: #f8f9fa;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Dental AI Platform") as demo:

    gr.HTML("""
        <div class="header-gradient">
            <h1 style="margin: 0; font-size: 2.8em;">🦷 Dental AI Platform</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                AI-powered dental analysis with Hugging Face dataset integration
            </p>
        </div>
    """)

    with gr.Tabs():

        # ============ TAB 1: WISDOM TOOTH DETECTION ============
        with gr.Tab("🔍 Wisdom Tooth Detection"):
            gr.Markdown("""
            ### Upload a dental X-ray to detect wisdom teeth locations
            The AI will analyze the image and draw bounding boxes around detected wisdom teeth.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    xray_input = gr.Image(type="pil", label="Upload Dental X-Ray", height=400)
                    model_selector = gr.Radio(
                        choices=["GPT-4o Vision", "Gemini Vision"],
                        value="GPT-4o Vision",
                        label="Select Vision Model"
                    )
                    analyze_btn = gr.Button("🔎 Analyze X-Ray", variant="primary", size="lg")

                with gr.Column(scale=1):
                    annotated_output = gr.Image(label="AI Analysis Result", height=400)
                    analysis_output = gr.Markdown(value="Results will appear here...")

            analyze_btn.click(
                fn=process_xray,
                inputs=[xray_input, model_selector],
                outputs=[annotated_output, analysis_output]
            )

        # ============ TAB 2: MULTI-MODEL CHATBOT ============
        with gr.Tab("💬 Multi-Model Chatbot"):
            gr.Markdown("""
            ### Ask a question and get responses from 3 AI models simultaneously
            Compare answers from OpenAI GPT-4o, Google Gemini, and Groq Llama3.
            """)

            with gr.Row():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything...",
                    lines=3,
                    scale=4
                )

            with gr.Row():
                submit_btn = gr.Button("🚀 Ask All Models", variant="primary", size="lg", scale=1)
                clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1)

            gr.Markdown("### 📊 Model Responses")

            with gr.Row():
                openai_output = gr.Markdown(value="*Response will appear here...*")
                gemini_output = gr.Markdown(value="*Response will appear here...*")
                groq_output = gr.Markdown(value="*Response will appear here...*")

            gr.Examples(
                examples=[
                    ["What are the symptoms of impacted wisdom teeth?"],
                    ["When should wisdom teeth be removed?"],
                    ["Explain dental X-ray interpretation basics"]
                ],
                inputs=query_input
            )

            submit_btn.click(
                fn=chat_with_all_models,
                inputs=[query_input],
                outputs=[openai_output, gemini_output, groq_output]
            )

            query_input.submit(
                fn=chat_with_all_models,
                inputs=[query_input],
                outputs=[openai_output, gemini_output, groq_output]
            )

            clear_btn.click(
                fn=lambda: ("", "*Response will appear here...*", "*Response will appear here...*", "*Response will appear here...*"),
                outputs=[query_input, openai_output, gemini_output, groq_output]
            )

        # ============ TAB 3: DATASET EXPLORER ============
        with gr.Tab("📊 Dataset Explorer"):
            gr.Markdown("""
            ### Explore the RayanAi Dental X-Ray Dataset (1,206 samples)
            Browse, search, and batch-process dental X-rays from Hugging Face.
            """)

            with gr.Row():
                load_dataset_btn = gr.Button("📥 Load Dataset from Hugging Face", variant="primary", size="lg")

            dataset_info = gr.Markdown(value="Click 'Load Dataset' to start...")

            load_dataset_btn.click(
                fn=load_hf_dataset,
                outputs=[dataset_info]
            )

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
            next_btn.click(
                fn=next_sample,
                inputs=[current_index],
                outputs=[sample_image, sample_info, current_index]
            )

            prev_btn.click(
                fn=prev_sample,
                inputs=[current_index],
                outputs=[sample_image, sample_info, current_index]
            )

            random_btn.click(
                fn=random_sample,
                outputs=[sample_image, sample_info, current_index]
            )

            jump_btn.click(
                fn=browse_sample,
                inputs=[jump_index],
                outputs=[sample_image, sample_info, current_index]
            )

            gr.Markdown("---\n### 🔄 Batch Processing")

            with gr.Row():
                batch_start = gr.Number(label="Start Index", value=0, minimum=0)
                batch_size = gr.Number(label="Batch Size", value=5, minimum=1, maximum=50)
                batch_model = gr.Dropdown(
                    choices=["GPT-4o Vision", "Gemini Vision"],
                    value="Gemini Vision",
                    label="Analysis Model"
                )

            batch_btn = gr.Button("⚡ Run Batch Analysis", variant="primary", size="lg")

            batch_output = gr.Markdown(value="Batch results will appear here...")

            batch_btn.click(
                fn=batch_process_samples,
                inputs=[batch_start, batch_size, batch_model],
                outputs=[batch_output]
            )

    gr.Markdown("""
    ---
    **Dental AI Platform** | Powered by OpenAI, Google, Groq, and Hugging Face
    """)


if __name__ == "__main__":
    print("🚀 Starting Enhanced Dental AI Platform...")
    print("📍 Server: http://localhost:7860")
    print("\nFeatures:")
    print("  - Tab 1: Wisdom Tooth Detection")
    print("  - Tab 2: Multi-Model Chatbot")
    print("  - Tab 3: Dataset Explorer (1,206 samples)")
    print("\n" + "="*50 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
