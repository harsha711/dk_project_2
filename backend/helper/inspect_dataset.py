#!/usr/bin/env python3
"""
Inspect dental X-ray dataset
Shows dataset statistics, class distribution, and sample images
"""
import os
import yaml
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt


def inspect_dataset(data_yaml_path: str = "Dental-X-ray-1/data.yaml"):
    """
    Inspect YOLO dataset structure and statistics
    
    Args:
        data_yaml_path: Path to dataset YAML file
    """
    if not os.path.exists(data_yaml_path):
        print(f"❌ Dataset not found: {data_yaml_path}")
        print("   Please download the dataset first or specify correct path")
        return
    
    print(f"📊 Inspecting dataset: {data_yaml_path}")
    print("=" * 60)
    
    # Load YAML config
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    dataset_dir = os.path.dirname(data_yaml_path)
    
    # Dataset info
    print(f"\n📁 Dataset Directory: {dataset_dir}")
    print(f"📝 Classes: {data.get('names', [])}")
    print(f"🔢 Number of classes: {data.get('nc', 0)}")
    
    # Count images and labels
    splits = ['train', 'val', 'test']
    total_images = 0
    total_labels = 0
    class_counts = Counter()
    
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            images_dir = os.path.join(split_path, 'images')
            labels_dir = os.path.join(split_path, 'labels')
            
            if os.path.exists(images_dir):
                num_images = len(list(Path(images_dir).glob('*.jpg')) + 
                               list(Path(images_dir).glob('*.png')))
                total_images += num_images
                print(f"\n  {split.upper()}:")
                print(f"    Images: {num_images}")
            
            if os.path.exists(labels_dir):
                num_labels = len(list(Path(labels_dir).glob('*.txt')))
                total_labels += num_labels
                print(f"    Labels: {num_labels}")
                
                # Count class occurrences
                for label_file in Path(labels_dir).glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                class_counts[class_id] += 1
    
    print(f"\n📈 Summary:")
    print(f"  Total Images: {total_images}")
    print(f"  Total Labels: {total_labels}")
    
    print(f"\n🏷️  Class Distribution:")
    for class_id, count in sorted(class_counts.items()):
        class_name = data.get('names', [])[class_id] if class_id < len(data.get('names', [])) else f"Class {class_id}"
        print(f"  {class_name}: {count} instances")
    
    # Sample image
    train_images = os.path.join(dataset_dir, 'train', 'images')
    if os.path.exists(train_images):
        sample_files = list(Path(train_images).glob('*.jpg'))[:3]
        if sample_files:
            print(f"\n🖼️  Sample Images:")
            for img_path in sample_files:
                img = Image.open(img_path)
                print(f"  {img_path.name}: {img.size[0]}x{img.size[1]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect YOLO dataset")
    parser.add_argument("--data", type=str, default="Dental-X-ray-1/data.yaml",
                        help="Path to dataset YAML file")
    
    args = parser.parse_args()
    
    inspect_dataset(args.data)

