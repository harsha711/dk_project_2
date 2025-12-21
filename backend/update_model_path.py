#!/usr/bin/env python3
"""
Update model path in api_utils.py
Helper script to change which YOLO model is used
"""
import os
import re
from pathlib import Path


def update_model_path(new_model_path: str, api_utils_path: str = "api_utils.py"):
    """
    Update YOLO model path in api_utils.py
    
    Args:
        new_model_path: New path to model file (relative to backend/ or absolute)
        api_utils_path: Path to api_utils.py file
    """
    if not os.path.exists(api_utils_path):
        print(f"❌ File not found: {api_utils_path}")
        return False
    
    # Read file
    with open(api_utils_path, 'r') as f:
        content = f.read()
    
    # Find and replace YOLO_IMPACTED_MODEL_PATH
    pattern = r'YOLO_IMPACTED_MODEL_PATH\s*=\s*[^\n]+'
    
    # Convert to absolute path if relative
    if not os.path.isabs(new_model_path):
        backend_dir = os.path.dirname(os.path.abspath(api_utils_path))
        new_model_path = os.path.join(backend_dir, new_model_path)
    
    # Normalize path
    new_model_path = os.path.normpath(new_model_path)
    
    replacement = f"YOLO_IMPACTED_MODEL_PATH = r\"{new_model_path}\""
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        
        # Write back
        with open(api_utils_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Updated model path to: {new_model_path}")
        return True
    else:
        print("❌ Could not find YOLO_IMPACTED_MODEL_PATH in file")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update YOLO model path in api_utils.py")
    parser.add_argument("model_path", type=str, help="Path to new model file")
    parser.add_argument("--file", type=str, default="api_utils.py", help="Path to api_utils.py")
    
    args = parser.parse_args()
    
    update_model_path(args.model_path, args.file)

