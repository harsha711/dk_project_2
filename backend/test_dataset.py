#!/usr/bin/env python3
"""
Test dataset loading and sample retrieval
Quick verification that dataset is properly configured
"""
from dataset_utils import TeethDatasetManager


def test_dataset():
    """Test dataset loading and sample retrieval"""
    print("🧪 Testing Dataset Manager...")
    print("=" * 60)
    
    manager = TeethDatasetManager()
    
    # Test loading
    print("\n1️⃣ Testing dataset loading...")
    result = manager.load_dataset()
    if result["success"]:
        print(f"   ✅ {result['message']}")
        print(f"   📊 Total samples: {result['total_samples']}")
    else:
        print(f"   ❌ {result['message']}")
        return False
    
    # Test getting a sample
    print("\n2️⃣ Testing sample retrieval...")
    sample = manager.get_sample(0)
    if sample and sample.get("image"):
        print(f"   ✅ Retrieved sample #0")
        print(f"   📏 Image size: {sample['image'].size}")
        print(f"   🏷️  Label: {sample.get('label', 'N/A')}")
    else:
        print("   ❌ Failed to retrieve sample")
        return False
    
    # Test random sample
    print("\n3️⃣ Testing random sample...")
    random_sample = manager.get_random_sample()
    if random_sample and random_sample.get("image"):
        print(f"   ✅ Retrieved random sample")
        print(f"   📏 Image size: {random_sample['image'].size}")
    else:
        print("   ❌ Failed to retrieve random sample")
        return False
    
    # Test dataset stats
    print("\n4️⃣ Testing dataset statistics...")
    stats = manager.get_dataset_stats()
    if stats:
        print(f"   ✅ Dataset stats retrieved")
        print(f"   📊 {stats}")
    else:
        print("   ⚠️  No stats available")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_dataset()

