"""
Multimodal utilities for unified chatbot interface
Handles message routing, context building, and response formatting

YOLO + Text Models:
- YOLOv8 for accurate bounding box detection
- GPT-4o-mini, Llama 3.3, and Mixtral 8x7B for text analysis
"""
from typing import Dict, List, Optional, Tuple
from PIL import Image


# System prompt for wisdom teeth specialist
SYSTEM_PROMPT = """You are a dental assistant specializing in wisdom teeth. You can analyze dental X-rays and answer follow-up questions about them. Stay focused on wisdom teeth topics.

IMPORTANT: You have access to the full conversation history. When users ask follow-up questions, use the previous conversation context to provide relevant answers. Remember what was discussed earlier in the conversation.

Your expertise includes:
- Wisdom tooth anatomy, development, and eruption patterns
- Common problems: impaction, pericoronitis, cysts, damage to adjacent teeth
- Dental X-ray interpretation (panoramic, periapical)
- Treatment options: monitoring, extraction, surgical considerations
- Post-operative care and recovery

Guidelines:
- Provide clear, helpful information about wisdom teeth
- Use conversation history to answer follow-up questions contextually
- Be specific about tooth positions (e.g., "upper right third molar #1", "lower left #17")
- When analyzing X-rays, describe: position, impaction angle, root formation, proximity to nerves
- Focus on wisdom teeth topics and stay on-topic
- If a user references something from earlier in the conversation, use that context in your response

IMPORTANT DISCLAIMER: This is for educational and informational purposes only. Not for clinical diagnosis. Real medical decisions must be made by licensed dental professionals."""


def route_message(
    message: str,
    image: Optional[Image.Image],
    history: List[Dict]
) -> Tuple[str, List[str]]:
    """
    Determine which models to use based on input type and context

    Args:
        message: User's text message
        image: Uploaded image (if any)
        history: Conversation history

    Returns:
        (mode, model_list) where mode is "vision", "vision-followup", or "chat"
    """
    try:
        has_image = image is not None
        has_recent_image = any(
            msg.get('image') is not None
            for msg in history[-3:] if msg.get('role') == 'user'
        )

        # Check if message references previous image
        image_refs = [
            'this x-ray', 'the image', 'this image', 'the x-ray',
            'in the picture', 'what you see', 'from the scan',
            'it', 'this', 'that'
        ]
        mentions_image = any(ref in message.lower() for ref in image_refs)

        if has_image:
            # User uploaded new image - use YOLO + text models for analysis
            # Vision models removed - they hallucinate coordinates
            mode, models = "vision", ["gpt4", "groq", "mixtral"]
        elif has_recent_image and mentions_image:
            # Follow-up question about previous image - use text models directly
            # They have YOLO results + conversation history for context
            mode, models = "chat", ["gpt4", "groq", "mixtral"]
        else:
            # Text-only question - use all text models
            mode, models = "chat", ["gpt4", "groq", "mixtral"]

        # Log routing decision
        print(f"[ROUTING] Mode: {mode}, Models: {models}, Has Image: {has_image}, Has Recent: {has_recent_image}")

        return mode, models

    except Exception as e:
        print(f"❌ ERROR in route_message: {e}")
        import traceback
        traceback.print_exc()
        # Default to chat mode on error
        return "chat", ["gpt4", "groq", "mixtral"]


def build_conversation_context(
    history: List[Dict],
    max_turns: int = 5,
    include_system_prompt: bool = True
) -> List[Dict]:
    """
    Build message context for API calls

    Args:
        history: Full conversation history
        max_turns: Maximum number of turns to include
        include_system_prompt: Whether to prepend system prompt

    Returns:
        List of messages formatted for API
    """
    messages = []

    # Add system prompt
    if include_system_prompt:
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })

    # Get recent history
    if history:
        recent_history = history[-(max_turns * 2):] if len(history) > max_turns * 2 else history
    else:
        recent_history = []

    print(f"[CONTEXT DEBUG] Building context from {len(history)} total messages")

    for msg in recent_history:
        if msg['role'] == 'user':
            user_msg = {"role": "user", "content": msg['content']}
            if msg.get('image'):
                user_msg['image'] = msg['image']
            messages.append(user_msg)

        elif msg['role'] == 'assistant':
            model_responses = msg.get('model_responses', {})

            # Get text responses (YOLO + text models only)
            text_response = (
                model_responses.get('gpt4', '') or
                model_responses.get('groq', '') or
                model_responses.get('mixtral', '')
            )

            # Add to context
            if text_response:
                messages.append({
                    "role": "assistant",
                    "content": text_response if isinstance(text_response, str) else str(text_response)
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": "[Previous response not available]"
                })

    print(f"  Final context has {len(messages)} messages")
    return messages


def format_multi_model_response(responses: Dict[str, str]) -> str:
    """
    Format multiple model responses for display in chatbot

    Args:
        responses: Dict mapping model names to response strings

    Returns:
        Formatted markdown string
    """
    gpt4 = responses.get('gpt4', 'No response')
    groq = responses.get('groq', 'No response')
    mixtral = responses.get('mixtral', 'No response')

    formatted = f"""### 🤖 AI Responses

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">

<div style="border: 2px solid #667eea; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #667eea;">🟢 GPT-4o-mini</h4>
{gpt4}
</div>

<div style="border: 2px solid #95E1D3; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #95E1D3;">🔵 Llama 3.3 70B</h4>
{groq}
</div>

<div style="border: 2px solid #F093FB; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #F093FB;">🟣 Mixtral 8x7B</h4>
{mixtral}
</div>

</div>
"""

    return formatted


def format_vision_response(
    responses: Dict[str, str],
    annotated_images: Optional[Dict[str, Image.Image]] = None
) -> str:
    """
    Format YOLO detection + text model analysis responses
    SIMPLIFIED: Vision models removed, only YOLO + text models

    Args:
        responses: Dict mapping model names to response text strings
        annotated_images: Dict mapping model names to annotated images

    Returns:
        Formatted markdown string with images
    """
    # Extract text model responses (GPT-4, Groq, and Mixtral analyze YOLO results)
    gpt4_text = responses.get('gpt4', '')
    groq_text = responses.get('groq', '')
    mixtral_text = responses.get('mixtral', '')

    has_text_models = bool(gpt4_text or groq_text or mixtral_text)

    if has_text_models:
        # YOLO detection + text model analysis (works for both initial and follow-up)
        formatted = f"""### 🦷 Dental Analysis

<div style="border: 2px solid #FF6B6B; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
<h4 style="margin-top: 0; color: #FF6B6B;">🎯 YOLOv8 Detection</h4>
<p>Accurate bounding box detection using trained dental AI model. See annotated X-ray below.</p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">

<div style="border: 2px solid #667eea; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #667eea;">🟢 GPT-4o-mini</h4>
{gpt4_text if gpt4_text else 'No response'}
</div>

<div style="border: 2px solid #95E1D3; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #95E1D3;">🔵 Llama 3.3 70B</h4>
{groq_text if groq_text else 'No response'}
</div>

<div style="border: 2px solid #F093FB; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #F093FB;">🟣 Mixtral 8x7B</h4>
{mixtral_text if mixtral_text else 'No response'}
</div>

</div>
"""
    else:
        # Fallback - no models available
        formatted = f"""### 🔍 Analysis

<div style="border: 2px solid #888; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #888;">⚠️ No Analysis Available</h4>
<p>Unable to generate analysis at this time. Please try again.</p>
</div>
"""

    # Add note about annotated images
    if annotated_images:
        num_annotated = len(annotated_images)
        formatted += f"\n\n**📊 Analysis Complete:** {num_annotated} detection(s) with bounding boxes shown below."

    return formatted

