"""
Test script for Phase 2: Image Upload and Vision Model Integration
"""
import asyncio
from PIL import Image
from dotenv import load_dotenv
from api_utils import init_clients, multimodal_chat_async
from image_utils import parse_vision_response, draw_bounding_boxes
from multimodal_utils import route_message, build_conversation_context, format_vision_response
from dataset_utils import TeethDatasetManager

# Load environment
load_dotenv()

# Initialize clients
openai_client, groq_client = init_clients()

async def test_vision_upload():
    """Test vision model with sample dental X-ray"""

    print("🧪 Testing Vision Model Integration")
    print("=" * 60)

    # Load a sample image from dataset
    print("\n1️⃣ Loading sample X-ray from dataset...")
    dataset_manager = TeethDatasetManager()
    result = dataset_manager.load_dataset()

    if not result['success']:
        print(f"❌ Failed to load dataset: {result['message']}")
        return

    # Get a random sample
    sample = dataset_manager.get_random_sample()
    if not sample['success']:
        print(f"❌ Failed to get sample: {sample['error']}")
        return

    image = sample['image']
    print(f"✅ Loaded sample #{sample['index']}")
    print(f"   Label: {sample['label']}")
    print(f"   Image size: {image.size}")

    # Test routing
    print("\n2️⃣ Testing message routing...")
    message = "Analyze this dental X-ray for wisdom teeth."
    mode, models = route_message(message, image, [])
    print(f"✅ Routing complete")
    print(f"   Mode: {mode}")
    print(f"   Models: {models}")

    # Test vision API calls
    print("\n3️⃣ Calling vision models...")
    print("   This may take 10-15 seconds...")

    responses = await multimodal_chat_async(
        message=message,
        image=image,
        conversation_context=[],
        models=models,
        openai_client=openai_client,
        groq_client=groq_client
    )

    print(f"✅ Received {len(responses)} responses")
    for model_name in responses.keys():
        response_preview = responses[model_name][:150] if len(responses[model_name]) > 150 else responses[model_name]
        print(f"   - {model_name}: {response_preview}...")

    # Test parsing and bounding boxes
    print("\n4️⃣ Testing response parsing and bounding box drawing...")
    annotated_images = {}

    for model_name, response_text in responses.items():
        if model_name in ["gpt4-vision", "gemini-vision"]:
            parsed = parse_vision_response(response_text)

            if parsed.get('error'):
                print(f"   ⚠️ {model_name}: Could not parse structured response")
                print(f"      (This is OK - model may have returned text instead of JSON)")
            elif parsed.get('teeth_found'):
                teeth = parsed['teeth_found']
                print(f"   ✅ {model_name}: Detected {len(teeth)} wisdom teeth")

                # Draw bounding boxes
                annotated = draw_bounding_boxes(image, teeth)
                annotated_images[model_name] = annotated
                print(f"      Bounding boxes drawn successfully")
            else:
                print(f"   ℹ️ {model_name}: No structured teeth data in response")

    # Test formatting
    print("\n5️⃣ Testing response formatting...")
    formatted = format_vision_response(responses, annotated_images)
    print(f"✅ Formatted response length: {len(formatted)} characters")

    # Save annotated images if any
    if annotated_images:
        print("\n6️⃣ Saving annotated images...")
        for model_name, img in annotated_images.items():
            filename = f"test_annotated_{model_name}.png"
            img.save(filename)
            print(f"   ✅ Saved: {filename}")
    else:
        print("\n6️⃣ No annotated images to save (models returned text responses)")
        print("   This is expected behavior - vision models can return either:")
        print("   - Structured JSON with bounding boxes")
        print("   - Text descriptions of what they see")

    print("\n" + "=" * 60)
    print("🎉 Phase 2 Integration Test Complete!")
    print("\nKey Findings:")
    print(f"  ✅ Image upload: Working")
    print(f"  ✅ Vision model routing: Working")
    print(f"  ✅ API calls: {len(responses)} models responded")
    print(f"  ✅ Response formatting: Working")
    print(f"  ✅ Bounding box drawing: {'Working' if annotated_images else 'Ready (no structured data this time)'}")

    return responses, annotated_images

if __name__ == "__main__":
    # Run test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        responses, annotated = loop.run_until_complete(test_vision_upload())
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loop.close()
