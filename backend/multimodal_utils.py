"""
Multimodal utilities for unified chatbot interface
Handles message routing, context building, and response formatting

FIXED VERSION:
- Groq Vision as primary (Gemini is rate-limited)
- Better handling of vision model fallbacks
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


def convert_vision_to_text(vision_responses: List[tuple]) -> str:
    """
    Convert vision model responses (JSON or natural language) to readable text for context
    
    Args:
        vision_responses: List of tuples (model_name, response_text)
    
    Returns:
        Natural language summary of vision analysis
    """
    summary_parts = []
    
    for model_name, vision_response in vision_responses:
        if not vision_response:
            continue
            
        try:
            import json
            # Try to parse as JSON
            parsed = json.loads(vision_response)
            if isinstance(parsed, dict):
                # Create a natural language summary from JSON
                model_summary = []
                if parsed.get('summary'):
                    model_summary.append(f"{parsed['summary']}")
                if parsed.get('teeth_found') and isinstance(parsed['teeth_found'], list):
                    teeth_count = len(parsed['teeth_found'])
                    if teeth_count > 0:
                        model_summary.append(f"Found {teeth_count} wisdom tooth/teeth:")
                        for tooth in parsed['teeth_found']:
                            pos = tooth.get('position', 'unknown')
                            desc = tooth.get('description', 'no description')
                            model_summary.append(f"- {pos}: {desc}")
                    else:
                        model_summary.append("No wisdom teeth detected.")
                
                if model_summary:
                    summary_parts.append(f"[{model_name}]: {' '.join(model_summary)}")
        except (json.JSONDecodeError, AttributeError, TypeError):
            # Not JSON or parsing failed - use as-is (might be natural language already)
            response_str = str(vision_response).lower()
            # Skip error/unavailable messages
            if "not currently available" not in response_str and "not available" not in response_str and "quota exceeded" not in response_str:
                summary_parts.append(f"[{model_name}]: {vision_response}")
    
    return "\n".join(summary_parts) if summary_parts else ""


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
            mode, models = "vision", ["gpt4", "groq"]
        elif has_recent_image and mentions_image:
            # Follow-up question about previous image - use text models directly
            # They have YOLO results + conversation history for context
            mode, models = "chat", ["gpt4", "groq"]
        else:
            # Text-only question - use both text models
            # Gemini removed (not implemented)
            mode, models = "chat", ["gpt4", "groq"]

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

            # Get text responses (vision models removed - using YOLO + text only)
            text_response = (
                model_responses.get('gpt4', '') or
                model_responses.get('groq', '') or
                model_responses.get('gemini', '')
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
    gemini = responses.get('gemini', 'No response')
    groq = responses.get('groq', 'No response')
    
    # Check if Gemini hit rate limit
    if "quota exceeded" in gemini.lower() or "429" in gemini:
        gemini = "⚠️ Gemini quota exceeded (free tier limit). See other responses."

    formatted = f"""### 🤖 AI Responses

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">

<div style="border: 2px solid #667eea; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #667eea;">🟢 GPT-4o-mini</h4>
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
    Format YOLO detection + text model analysis responses
    SIMPLIFIED: Vision models removed, only YOLO + text models

    Args:
        responses: Dict mapping model names to response text strings
        annotated_images: Dict mapping model names to annotated images

    Returns:
        Formatted markdown string with images
    """
    # Extract text model responses (GPT-4 and Groq analyze YOLO results)
    gpt4_text = responses.get('gpt4', '')
    groq_text = responses.get('groq', '')

    has_text_models = bool(gpt4_text or groq_text)

    if has_text_models:
        # YOLO detection + text model analysis (works for both initial and follow-up)
        formatted = f"""### 🦷 Dental Analysis

<div style="border: 2px solid #FF6B6B; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
<h4 style="margin-top: 0; color: #FF6B6B;">🎯 YOLOv8 Detection</h4>
<p>Accurate bounding box detection using trained dental AI model. See annotated X-ray below.</p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">

<div style="border: 2px solid #667eea; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #667eea;">🟢 GPT-4o-mini Analysis</h4>
{gpt4_text if gpt4_text else 'No response'}
</div>

<div style="border: 2px solid #95E1D3; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #95E1D3;">🔵 Groq Llama 3.3 70B</h4>
{groq_text if groq_text else 'No response'}
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


def extract_teeth_summary(responses: Dict[str, str]) -> str:
    """
    Extract key findings from multiple model responses
    """
    import re
    all_text = ' '.join(str(v) for v in responses.values()).lower()

    findings = []

    if 'impacted' in all_text or 'impaction' in all_text:
        findings.append("Impacted wisdom teeth detected")

    if 'pain' in all_text or 'painful' in all_text:
        findings.append("Pain indicators present")

    if 'infection' in all_text or 'pericoronitis' in all_text:
        findings.append("Possible infection signs")

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
    """
    wisdom_keywords = [
        'wisdom', 'third molar', 'impacted', 'extraction',
        'tooth 1', 'tooth 16', 'tooth 17', 'tooth 32',
        '#1', '#16', '#17', '#32'
    ]

    message_lower = message.lower()
    is_wisdom_related = any(keyword in message_lower for keyword in wisdom_keywords)

    general_dental = any(word in message_lower for word in [
        'cavity', 'crown', 'root canal', 'filling', 'braces',
        'whitening', 'cleaning', 'gingivitis'
    ])

    if general_dental and not is_wisdom_related:
        return False, "I specialize in wisdom teeth. For that dental concern, please consult a general dentist."

    return True, None