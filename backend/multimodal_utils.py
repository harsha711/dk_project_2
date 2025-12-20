"""
Multimodal utilities for unified chatbot interface
Handles message routing, context building, and response formatting
"""
from typing import Dict, List, Optional, Tuple
from PIL import Image
import re


# System prompt for wisdom teeth specialist
SYSTEM_PROMPT = """You are a specialized dental AI assistant focused on wisdom teeth (third molars).

Your expertise includes:
- Wisdom tooth anatomy, development, and eruption patterns
- Common problems: impaction, pericoronitis, cysts, damage to adjacent teeth
- Dental X-ray interpretation (panoramic, periapical)
- Treatment options: monitoring, extraction, surgical considerations
- Post-operative care and recovery

Guidelines:
- Answer ONLY wisdom teeth related questions
- Be specific about tooth positions (e.g., "upper right third molar #1", "lower left #17")
- When analyzing X-rays, describe: position, impaction angle, root formation, proximity to nerves
- For other dental topics, politely redirect: "I specialize in wisdom teeth. Please consult a general dentist for that concern."
- For medical emergencies (severe pain, infection, bleeding), advise seeking immediate care
- Base responses on clinical evidence when possible

Remember: You assist with information only. All clinical decisions should be made by licensed dental professionals."""


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
            # User uploaded new image - use vision models
            mode, models = "vision", ["gpt4-vision", "gemini-vision"]
        elif has_recent_image and mentions_image:
            # Follow-up question about previous image
            mode, models = "vision-followup", ["gpt4-vision", "gemini-vision"]
        else:
            # Text-only question - use all chat models
            mode, models = "chat", ["gpt4", "gemini", "groq"]

        # Log routing decision
        print(f"[ROUTING] Mode: {mode}, Models: {models}, Has Image: {has_image}, Has Recent: {has_recent_image}")

        return mode, models

    except Exception as e:
        print(f"❌ ERROR in route_message: {e}")
        import traceback
        traceback.print_exc()
        # Default to chat mode on error
        return "chat", ["gpt4", "gemini", "groq"]


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

    # Get recent history (last N user-assistant pairs)
    recent_history = history[-(max_turns * 2):] if history else []

    for msg in recent_history:
        if msg['role'] == 'user':
            user_msg = {"role": "user", "content": msg['content']}

            # Include image if present (for vision models)
            if msg.get('image'):
                user_msg['image'] = msg['image']

            messages.append(user_msg)

        elif msg['role'] == 'assistant':
            # Use primary model's response for context
            # Prefer gpt4-vision for vision responses, gpt4 for text
            model_responses = msg.get('model_responses', {})
            primary_response = (
                model_responses.get('gpt4-vision', '') or
                model_responses.get('gpt4', '') or
                model_responses.get('gemini-vision', '') or
                model_responses.get('gemini', '')
            )
            if primary_response:
                messages.append({
                    "role": "assistant",
                    "content": primary_response
                })

    return messages


def format_multi_model_response(responses: Dict[str, str]) -> str:
    """
    Format multiple model responses for display in chatbot

    Args:
        responses: Dict mapping model names to response strings

    Returns:
        Formatted markdown string
    """
    # Extract responses
    gpt4 = responses.get('gpt4', 'No response')
    gemini = responses.get('gemini', 'No response')
    groq = responses.get('groq', 'No response')

    # Format as 3-column markdown table
    formatted = f"""### 🤖 AI Responses

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">

<div style="border: 2px solid #667eea; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #667eea;">🟢 GPT-4o</h4>
{gpt4}
</div>

<div style="border: 2px solid #4ECDC4; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #4ECDC4;">🔵 Gemini</h4>
{gemini}
</div>

<div style="border: 2px solid #FF6B6B; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #FF6B6B;">🟠 Groq Llama3</h4>
{groq}
</div>

</div>
"""

    return formatted


def format_vision_response(
    responses: Dict[str, str],
    annotated_images: Optional[Dict[str, Image.Image]] = None
) -> str:
    """
    Format vision model responses with annotated images

    Args:
        responses: Dict mapping model names to response text strings
        annotated_images: Dict mapping model names to annotated images

    Returns:
        Formatted markdown string with images
    """
    # Handle both dict and string responses
    if isinstance(responses.get('gpt4-vision'), dict):
        gpt4_text = responses.get('gpt4-vision', {}).get('response', 'No response')
        gemini_text = responses.get('gemini-vision', {}).get('response', 'No response')
    else:
        gpt4_text = responses.get('gpt4-vision', 'No response')
        gemini_text = responses.get('gemini-vision', 'No response')

    formatted = f"""### 🔍 Vision Analysis

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">

<div style="border: 2px solid #667eea; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #667eea;">🟢 GPT-4o Vision</h4>
{gpt4_text}
</div>

<div style="border: 2px solid #4ECDC4; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #4ECDC4;">🔵 Gemini Vision</h4>
{gemini_text}
</div>

</div>
"""

    # Add note about annotated images if they exist
    if annotated_images:
        num_annotated = len(annotated_images)
        formatted += f"\n\n**📊 Analysis Complete:** {num_annotated} model(s) detected wisdom teeth with bounding boxes."

    return formatted


def extract_teeth_summary(responses: Dict[str, str]) -> str:
    """
    Extract key findings from multiple model responses
    Creates a consensus summary

    Args:
        responses: Model responses

    Returns:
        Brief summary of key findings
    """
    # Simple extraction - can be enhanced with NLP
    all_text = ' '.join(responses.values()).lower()

    findings = []

    # Count mentions
    if 'impacted' in all_text or 'impaction' in all_text:
        findings.append("Impacted wisdom teeth detected")

    if 'pain' in all_text or 'painful' in all_text:
        findings.append("Pain indicators present")

    if 'infection' in all_text or 'pericoronitis' in all_text:
        findings.append("Possible infection signs")

    # Extract numbers
    tooth_counts = re.findall(r'(\d+)\s+wisdom\s+teeth', all_text)
    if tooth_counts:
        most_common = max(set(tooth_counts), key=tooth_counts.count)
        findings.insert(0, f"{most_common} wisdom teeth identified")

    if findings:
        return "**Key Findings:** " + " | ".join(findings)
    else:
        return ""


def validate_dental_question(message: str) -> Tuple[bool, Optional[str]]:
    """
    Check if question is wisdom teeth related

    Args:
        message: User's question

    Returns:
        (is_valid, redirect_message)
    """
    # Keywords that indicate wisdom teeth topic
    wisdom_keywords = [
        'wisdom', 'third molar', 'impacted', 'extraction',
        'tooth 1', 'tooth 16', 'tooth 17', 'tooth 32',  # Wisdom tooth numbers
        '#1', '#16', '#17', '#32'
    ]

    # Check if message contains wisdom teeth keywords
    message_lower = message.lower()
    is_wisdom_related = any(keyword in message_lower for keyword in wisdom_keywords)

    # If it's a general dental question (but not wisdom teeth)
    general_dental = any(word in message_lower for word in [
        'cavity', 'crown', 'root canal', 'filling', 'braces',
        'whitening', 'cleaning', 'gingivitis'
    ])

    if general_dental and not is_wisdom_related:
        return False, "I specialize in wisdom teeth. For that dental concern, please consult a general dentist."

    # Allow image uploads without keywords (they might be X-rays)
    # Allow follow-up questions in ongoing conversations
    return True, None


def format_chat_message_for_display(
    role: str,
    content: str,
    image: Optional[Image.Image] = None,
    model_responses: Optional[Dict] = None
) -> Tuple[str, Optional[str]]:
    """
    Format a message for Gradio chatbot display

    Args:
        role: "user" or "assistant"
        content: Message text
        image: Attached image if any
        model_responses: Dict of model responses (for assistant)

    Returns:
        (formatted_content, image_path_or_none)
    """
    if role == "user":
        # User messages are simple
        if image:
            return content, image  # Gradio will handle image display
        else:
            return content, None

    elif role == "assistant":
        # Assistant messages need multi-model formatting
        if model_responses:
            formatted = format_multi_model_response(model_responses)
            return formatted, None
        else:
            return content, None

    return content, None


def truncate_long_response(text: str, max_chars: int = 1000) -> str:
    """
    Truncate overly long responses for better UX

    Args:
        text: Response text
        max_chars: Maximum characters

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_chars:
        return text

    # Truncate at sentence boundary if possible
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')

    if last_period > max_chars * 0.8:  # If we're close to limit
        return truncated[:last_period + 1] + "\n\n[Response truncated...]"
    else:
        return truncated + "...\n\n[Response truncated...]"
