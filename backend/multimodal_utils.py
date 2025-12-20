"""
Multimodal utilities for unified chatbot interface
Handles message routing, context building, and response formatting
"""
from typing import Dict, List, Optional, Tuple
from PIL import Image
import re


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
            # Check if it's an error message or unavailable message
            response_str = str(vision_response).lower()
            if "not currently available" not in response_str and "not available" not in response_str:
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
            # User uploaded new image - use vision models (Gemini Vision only, GPT-4o Vision removed due to refusals)
            mode, models = "vision", ["gemini-vision", "groq-vision"]
        elif has_recent_image and mentions_image:
            # Follow-up question about previous image
            # Use Gemini Vision to re-analyze, then GPT-4o-mini and Groq use Gemini's analysis as context
            mode, models = "vision-followup", ["gemini-vision", "gpt4", "groq"]
        else:
            # Text-only question - use all chat models (including Groq)
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
    # Include all messages up to max_turns pairs (user + assistant = 2 messages per turn)
    # Always include the last message even if it's unpaired (user message without assistant response)
    if history:
        # Get last max_turns * 2 messages, but ensure we include the very last message
        # This handles cases where the current user message hasn't been responded to yet
        recent_history = history[-(max_turns * 2):] if len(history) > max_turns * 2 else history
    else:
        recent_history = []

    # Debug: Print context building info
    print(f"[CONTEXT DEBUG] Building context from {len(history)} total messages")
    print(f"  Recent history: {len(recent_history)} messages")
    if recent_history:
        print(f"  First message: role={recent_history[0].get('role')}, preview={str(recent_history[0].get('content', ''))[:60]}...")
        print(f"  Last message: role={recent_history[-1].get('role')}, preview={str(recent_history[-1].get('content', ''))[:60]}...")

    for msg in recent_history:
        if msg['role'] == 'user':
            user_msg = {"role": "user", "content": msg['content']}

            # Include image if present (for vision models)
            if msg.get('image'):
                user_msg['image'] = msg['image']

            messages.append(user_msg)

        elif msg['role'] == 'assistant':
            # Use primary model's response for context
            # For GPT-4o-mini context, prefer GPT-4o-mini's own response, but fallback to any available response
            model_responses = msg.get('model_responses', {})
            
            # Debug: Print what model responses are available
            if model_responses:
                available_keys = list(model_responses.keys())
                print(f"  [CONTEXT] Assistant message has responses from: {available_keys}")
            
            # For text models (GPT-4o-mini, Groq), we need to include vision analysis in context
            # Priority: text responses > vision responses (converted to natural language)
            # Check for both gemini-vision and groq-vision responses
            
            # First, try to get text responses (best for context)
            text_response = (
                model_responses.get('gpt4', '') or
                model_responses.get('gemini', '') or
                model_responses.get('groq', '')
            )
            
            # Check for vision responses (gemini-vision or groq-vision)
            vision_responses = []
            if model_responses.get('gemini-vision'):
                vision_responses.append(('Gemini Vision', model_responses.get('gemini-vision')))
            if model_responses.get('groq-vision'):
                vision_responses.append(('Groq Vision', model_responses.get('groq-vision')))
            
            # If we have text responses, use them as primary
            if text_response:
                primary_response = text_response
                # Also append vision analysis if available (for better context)
                # This ensures text models see the vision analysis even if there are text responses
                if vision_responses:
                    vision_summary = convert_vision_to_text(vision_responses)
                    if vision_summary:
                        primary_response = f"{text_response}\n\n[Previous Vision Analysis: {vision_summary}]"
                        print(f"  [CONTEXT] Added vision analysis to text response context")
            elif vision_responses:
                # Only have vision responses - convert to natural language
                # This is the case for initial image analysis
                primary_response = convert_vision_to_text(vision_responses)
                if primary_response:
                    print(f"  [CONTEXT] Using vision analysis as context (converted to natural language)")
                else:
                    primary_response = None
            else:
                # No responses available
                primary_response = None
            
            # Always include assistant response if available (even if empty, to maintain structure)
            # This ensures conversation flow is preserved
            if primary_response:
                # Check if response is a string (should be)
                if isinstance(primary_response, str):
                    messages.append({
                        "role": "assistant",
                        "content": primary_response
                    })
                else:
                    # If it's a dict or other type, convert to string
                    print(f"  ⚠️ Warning: Assistant response is not a string, converting: {type(primary_response)}")
                    messages.append({
                        "role": "assistant",
                        "content": str(primary_response)
                    })
            else:
                # If no response found, log warning but don't skip (maintains message order)
                print(f"  ⚠️ Warning: Assistant message found but no model response available")
                print(f"     Model responses dict: {model_responses}")
                # Still add empty assistant message to maintain conversation structure
                messages.append({
                    "role": "assistant",
                    "content": "[Previous response not available]"
                })

    # Debug: Print final context
    print(f"  Final context has {len(messages)} messages (including system prompt)")
    for i, msg in enumerate(messages):
        if msg.get('role') == 'system':
            print(f"    [{i}] SYSTEM: {str(msg.get('content', ''))[:60]}...")
        else:
            print(f"    [{i}] {msg.get('role', 'unknown').upper()}: {str(msg.get('content', ''))[:60]}...")

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
    Format vision model responses with annotated images
    Handles both initial vision analysis and follow-up questions

    Args:
        responses: Dict mapping model names to response text strings
        annotated_images: Dict mapping model names to annotated images

    Returns:
        Formatted markdown string with images
    """
    # Handle both dict and string responses (GPT-4o Vision removed)
    if isinstance(responses.get('gemini-vision'), dict):
        gemini_vision_text = responses.get('gemini-vision', {}).get('response', 'No response')
        groq_vision_text = responses.get('groq-vision', {}).get('response', 'No response')
    else:
        gemini_vision_text = responses.get('gemini-vision', 'No response')
        groq_vision_text = responses.get('groq-vision', 'No response')
    
    # Check if we have text model responses (for follow-up questions)
    gpt4_text = responses.get('gpt4', '')
    groq_text = responses.get('groq', '')
    
    # Determine if this is a follow-up (has text models) or initial analysis (only vision models)
    has_text_models = bool(gpt4_text or groq_text)
    
    if has_text_models:
        # Follow-up question: Show Gemini Vision analysis + GPT-4o-mini + Groq responses
        formatted = f"""### 🔍 Follow-up Analysis (Based on Gemini Vision's Findings)

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">

<div style="border: 2px solid #4ECDC4; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #4ECDC4;">🔵 Gemini Vision</h4>
{gemini_vision_text if gemini_vision_text else 'Re-analyzing...'}
</div>

<div style="border: 2px solid #667eea; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #667eea;">🟢 GPT-4o-mini</h4>
{gpt4_text if gpt4_text else 'No response'}
</div>

<div style="border: 2px solid #FF6B6B; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #FF6B6B;">🟠 Groq Llama3</h4>
{groq_text if groq_text else 'No response'}
</div>

</div>

<p><em>💡 GPT-4o-mini and Groq are answering based on Gemini Vision's analysis of the X-ray.</em></p>
"""
    else:
        # Initial analysis: Gemini Vision (and Groq Vision if available)
        # Check if Groq Vision is available or just showing unavailable message
        groq_available = groq_vision_text and "not currently available" not in groq_vision_text.lower() and "not available" not in groq_vision_text.lower()
        
        if groq_available:
            # Show both Gemini and Groq Vision
            formatted = f"""### 🔍 Vision Analysis

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">

<div style="border: 2px solid #4ECDC4; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #4ECDC4;">🔵 Gemini Vision</h4>
{gemini_vision_text}
</div>

<div style="border: 2px solid #FF6B6B; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #FF6B6B;">🟠 Llama 3.2 Vision</h4>
{groq_vision_text}
</div>

</div>
"""
        else:
            # Only show Gemini Vision (Groq not available)
            groq_message = ""
            if groq_vision_text and ("not currently available" in groq_vision_text.lower() or "not available" in groq_vision_text.lower()):
                # Parse the JSON message from Groq
                try:
                    import json
                    groq_json = json.loads(groq_vision_text)
                    groq_message = f'<p style="color: #888; font-style: italic; margin-top: 10px;">ℹ️ {groq_json.get("summary", groq_vision_text)}</p>'
                except:
                    groq_message = f'<p style="color: #888; font-style: italic; margin-top: 10px;">ℹ️ {groq_vision_text}</p>'
            
            formatted = f"""### 🔍 Vision Analysis

<div style="border: 2px solid #4ECDC4; border-radius: 8px; padding: 12px;">
<h4 style="margin-top: 0; color: #4ECDC4;">🔵 Gemini Vision</h4>
{gemini_vision_text}
</div>

{groq_message}
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
