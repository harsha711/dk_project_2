#!/usr/bin/env python3
"""
Scan dataset quality - check how many images are valid X-rays vs binary masks
"""
from dataset_utils import get_teeth_dataset_manager

def main():
    print("=" * 60)
    print("Dataset Quality Scanner")
    print("=" * 60)
    
    manager = get_teeth_dataset_manager()
    
    # Scan quality
    result = manager.scan_dataset_quality(num_samples=100)
    
    if result.get("success"):
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(result["message"])
        print("\nRecommendation:")
        if result["valid_percent"] < 50:
            print("⚠️  Dataset has low quality - consider using a different dataset")
        elif result["valid_percent"] < 80:
            print("⚠️  Dataset has moderate quality - filtering will help")
        else:
            print("✅ Dataset quality is good")
    else:
        print(f"❌ Error: {result.get('message')}")

if __name__ == "__main__":
    main()

