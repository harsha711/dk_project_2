"""
Dental AI Platform - Gradio Application
Two tabs: Wisdom Tooth Detection & Multi-Model Chatbot
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
    create_side_by_side_comparison
)

# Load environment variables
load_dotenv()

# Initialize API clients globally
openai_client, groq_client = init_clients()


# ============ TAB 1: WISDOM TOOTH DETECTION ============

def process_xray(image: Image.Image, model_choice: str) -> tuple:
    """
    Process dental X-ray image with selected vision model

    Args:
        image: PIL Image of dental X-ray
        model_choice: "GPT-4V" or "Gemini Vision"

    Returns:
        (annotated_image, analysis_text)
    """
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
    """
    Send query to all 3 models in parallel and return responses

    Args:
        query: User's text query

    Returns:
        (openai_response, gemini_response, groq_response)
    """
    if not query or not query.strip():
        empty_msg = "Please enter a question first."
        return empty_msg, empty_msg, empty_msg

    try:
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            chat_all_models(query, openai_client, groq_client)
        )
        loop.close()

        # Unpack results
        openai_result, gemini_result, groq_result = results

        # Format responses with status indicators
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


# ============ GRADIO UI ============

# Custom CSS
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

.xray-container {
    border: 2px solid #764ba2;
    border-radius: 8px;
    padding: 10px;
}
"""

# Create Gradio app with tabs
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Dental AI Platform") as demo:

    # Header
    gr.HTML("""
        <div class="header-gradient">
            <h1 style="margin: 0; font-size: 2.8em;">🦷 Dental AI Platform</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                AI-powered dental analysis and multi-model chatbot
            </p>
        </div>
    """)

    # Create tabs
    with gr.Tabs():

        # ============ TAB 1: WISDOM TOOTH DETECTION ============
        with gr.Tab("🔍 Wisdom Tooth Detection"):
            gr.Markdown("""
            ### Upload a dental X-ray to detect wisdom teeth locations
            The AI will analyze the image and draw bounding boxes around detected wisdom teeth.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    # Input components
                    xray_input = gr.Image(
                        type="pil",
                        label="Upload Dental X-Ray",
                        height=400
                    )

                    model_selector = gr.Radio(
                        choices=["GPT-4o Vision", "Gemini Vision"],
                        value="GPT-4o Vision",
                        label="Select Vision Model",
                        info="Choose which AI model to use for analysis"
                    )

                    analyze_btn = gr.Button(
                        "🔎 Analyze X-Ray",
                        variant="primary",
                        size="lg"
                    )

                    gr.Markdown("""
                    **Supported models:**
                    - **GPT-4o Vision**: OpenAI's latest vision model
                    - **Gemini Vision**: Google's multimodal AI

                    **Instructions:**
                    1. Upload a dental X-ray image
                    2. Select your preferred AI model
                    3. Click "Analyze X-Ray" to detect wisdom teeth
                    """)

                with gr.Column(scale=1):
                    # Output components
                    annotated_output = gr.Image(
                        label="AI Analysis Result",
                        height=400,
                        elem_classes=["xray-container"]
                    )

                    analysis_output = gr.Markdown(
                        label="Analysis Details",
                        value="Results will appear here after analysis..."
                    )

            # Connect event
            analyze_btn.click(
                fn=process_xray,
                inputs=[xray_input, model_selector],
                outputs=[annotated_output, analysis_output]
            )

        # ============ TAB 2: MULTI-MODEL CHATBOT ============
        with gr.Tab("💬 Multi-Model Chatbot"):
            gr.Markdown("""
            ### Ask a question and get responses from 3 AI models simultaneously
            Compare answers from OpenAI GPT-4o, Google Gemini, and Groq Llama3 side by side.
            """)

            # Query input
            with gr.Row():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything... (e.g., 'What causes wisdom tooth pain?')",
                    lines=3,
                    scale=4
                )

            with gr.Row():
                submit_btn = gr.Button(
                    "🚀 Ask All Models",
                    variant="primary",
                    size="lg",
                    scale=1
                )
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    size="lg",
                    scale=1
                )

            # Response columns
            gr.Markdown("### 📊 Model Responses")

            with gr.Row():
                openai_output = gr.Markdown(
                    label="OpenAI GPT-4o",
                    value="*Response will appear here...*",
                    elem_classes=["response-box"]
                )
                gemini_output = gr.Markdown(
                    label="Google Gemini",
                    value="*Response will appear here...*",
                    elem_classes=["response-box"]
                )
                groq_output = gr.Markdown(
                    label="Groq Llama3",
                    value="*Response will appear here...*",
                    elem_classes=["response-box"]
                )

            # Examples
            gr.Examples(
                examples=[
                    ["What are the symptoms of impacted wisdom teeth?"],
                    ["When should wisdom teeth be removed?"],
                    ["What are the risks of wisdom tooth extraction?"],
                    ["How long is the recovery after wisdom tooth removal?"],
                    ["Explain the difference between machine learning and deep learning"]
                ],
                inputs=query_input,
                label="Example Questions"
            )

            # Connect events
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
                inputs=[],
                outputs=[query_input, openai_output, gemini_output, groq_output]
            )

    # Footer
    gr.Markdown("""
    ---
    **Dental AI Platform** | Powered by OpenAI, Google, and Groq APIs
    """)


# ============ LAUNCH APP ============

if __name__ == "__main__":
    print("🚀 Starting Dental AI Platform...")
    print("📍 Server will be available at: http://localhost:7860")
    print("\nMake sure your .env file contains:")
    print("  - OPEN_AI_API_KEY")
    print("  - GROQ_AI_API_KEY")
    print("  - GOOGLE_AI_API_KEY")
    print("\n" + "="*50 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False
    )
