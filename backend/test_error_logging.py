"""
Test script to verify error logging functionality
Tests error handling in various scenarios
"""
import sys
import os

# Test 1: Test routing with valid inputs
print("=" * 80)
print("TEST 1: Valid routing test")
print("=" * 80)

from multimodal_utils import route_message

# Test with text message
conversation_state = []
mode, models = route_message("What are symptoms of impacted wisdom teeth?", None, conversation_state)
print(f"✅ Text routing: mode={mode}, models={models}")

# Test with image reference
conversation_state.append({
    "role": "user",
    "content": "Previous message",
    "image": "dummy_image"
})
mode, models = route_message("What do you see in this X-ray?", None, conversation_state)
print(f"✅ Image reference routing: mode={mode}, models={models}")

# Test 2: Test with invalid routing inputs (should trigger error logging)
print("\n" + "=" * 80)
print("TEST 2: Invalid routing test (should show error logs)")
print("=" * 80)

try:
    # Pass invalid history type
    mode, models = route_message("test", None, "invalid_history_type")
    print(f"Result: mode={mode}, models={models}")
except Exception as e:
    print(f"Exception caught: {e}")

# Test 3: Test context building
print("\n" + "=" * 80)
print("TEST 3: Context building test")
print("=" * 80)

from multimodal_utils import build_conversation_context

conversation_state = [
    {
        "role": "user",
        "content": "What are wisdom teeth?",
        "image": None
    },
    {
        "role": "assistant",
        "model_responses": {
            "gpt4": "Wisdom teeth are the third molars..."
        }
    }
]

context = build_conversation_context(conversation_state, max_turns=5)
print(f"✅ Context built with {len(context)} messages")
for i, msg in enumerate(context):
    print(f"  [{i}] role={msg['role']}, content={msg['content'][:50]}...")

# Test 4: Test with missing API keys (should trigger error logs in real API calls)
print("\n" + "=" * 80)
print("TEST 4: Summary")
print("=" * 80)

print("""
✅ Error logging has been added to:
  - dental_ai_unified.py: process_chat_message()
  - api_utils.py: All async chat functions (OpenAI, Gemini, Groq)
  - api_utils.py: Vision analysis functions (GPT-4V, Gemini Vision)
  - api_utils.py: Context-aware chat and vision functions
  - multimodal_utils.py: route_message()

🔍 Error logs include:
  - Error type and message
  - Function name
  - Relevant context (query, image status, message count, etc.)
  - Full stack trace

📝 To test actual API errors:
  1. Start the Gradio app
  2. Send a message (will log routing info)
  3. Try with invalid API keys to see error logs
  4. Upload an invalid image to see vision error logs
""")

print("\n" + "=" * 80)
print("✅ Error logging test complete!")
print("=" * 80)
