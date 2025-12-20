"""
Quick test to verify message format compatibility
"""
import asyncio
from api_utils import init_clients, chat_with_context_async
from multimodal_utils import build_conversation_context, SYSTEM_PROMPT

async def test_text_chat():
    """Test basic text chat with context"""
    print("🧪 Testing Text Chat Format")
    print("=" * 60)

    # Initialize clients
    openai_client, groq_client = init_clients()

    # Simulate conversation state (Phase 3 format)
    conversation_state = [
        {
            "role": "user",
            "content": "What are symptoms of wisdom tooth pain?",
            "image": None,
            "timestamp": 1703088421.0
        }
    ]

    # Build context
    print("\n1️⃣ Building conversation context...")
    context = build_conversation_context(conversation_state, max_turns=5)
    print(f"✅ Context built: {len(context)} messages")
    for i, msg in enumerate(context):
        role = msg['role']
        content_preview = str(msg['content'])[:60] if 'content' in msg else ''
        has_image = 'image' in msg
        print(f"   [{i}] {role}: {content_preview}... (image: {has_image})")

    # Test chat API call
    print("\n2️⃣ Testing GPT-4 text chat...")
    try:
        result = await chat_with_context_async(
            messages=context,
            model_name="gpt4",
            openai_client=openai_client,
            groq_client=groq_client
        )

        if result.get('success'):
            print(f"✅ GPT-4 response received")
            print(f"   Response preview: {result['response'][:100]}...")
        else:
            print(f"❌ GPT-4 error: {result['response']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 Text chat format test complete!")

if __name__ == "__main__":
    asyncio.run(test_text_chat())
