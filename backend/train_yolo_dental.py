#!/usr/bin/env python3
"""
Train YOLO model on dental X-ray dataset
Supports both Roboflow and local dataset formats
"""
import os
from ultralytics import YOLO
from pathlib import Path


def train_yolo_model(
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    data_path: str = None,
    project_name: str = "dental_detection"
):
    """
    Train YOLO model on dental X-ray dataset
    
    Args:
        model_size: YOLO model size (nano, small, medium, large, xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        data_path: Path to dataset.yaml file (if None, uses default)
        project_name: Project name for saving results
    """
    # Model selection
    model_name = f"yolov8{model_size}.pt"
    print(f"🚀 Starting YOLO training with {model_name}")
    print(f"   Epochs: {epochs}, Image Size: {imgsz}, Batch: {batch}")
    
    # Initialize model
    model = YOLO(model_name)  # Automatically downloads if not present
    
    # Dataset path
    if data_path is None:
        # Check for common dataset locations
        possible_paths = [
            "Dental-X-ray-1/data.yaml",
            "datasets/Dental-X-ray-1/data.yaml",
            "data.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            print("❌ Error: No dataset found. Please specify data_path or place data.yaml in one of:")
            for path in possible_paths:
                print(f"   - {path}")
            return None
    
    print(f"📊 Using dataset: {data_path}")
    
    # Train model
    try:
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project_name,
            name="dental_wisdom_detection",
            patience=50,  # Early stopping patience
            save=True,
            plots=True,
            val=True,
            device=0,  # Use GPU 0 if available, else CPU
        )
        
        print(f"\n✅ Training completed!")
        print(f"📁 Results saved to: {results.save_dir}")
        print(f"🎯 Best model: {results.save_dir}/weights/best.pt")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO model on dental X-ray dataset")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="Model size: n (nano), s (small), m (medium), l (large), x (xlarge)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset.yaml file")
    parser.add_argument("--project", type=str, default="dental_detection", help="Project name")
    
    args = parser.parse_args()
    
    train_yolo_model(
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        data_path=args.data,
        project_name=args.project
    )

