#!/usr/bin/env python3
"""
Download dental X-ray dataset from Roboflow
Requires Roboflow API key in environment or .env file
"""
import os
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def download_roboflow_dataset(
    workspace: str = "dental-xray",
    project: str = "dental-x-ray-1",
    version: int = 1,
    api_key: str = None
):
    """
    Download dataset from Roboflow
    
    Args:
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version number
        api_key: Roboflow API key (if None, uses ROBOWFLOW_API_KEY env var)
    """
    if api_key is None:
        api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if not api_key:
        print("❌ Error: Roboflow API key not found!")
        print("   Set ROBOWFLOW_API_KEY environment variable or add to .env file")
        print("   Get your API key from: https://app.roboflow.com/")
        return False
    
    try:
        print(f"🔗 Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)
        
        print(f"📥 Downloading dataset: {workspace}/{project} (v{version})")
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8")
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"📁 Location: {dataset.location}")
        print(f"📄 Config: {dataset.location}/data.yaml")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download dental X-ray dataset from Roboflow")
    parser.add_argument("--workspace", type=str, default="dental-xray", help="Roboflow workspace")
    parser.add_argument("--project", type=str, default="dental-x-ray-1", help="Project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key")
    
    args = parser.parse_args()
    
    download_roboflow_dataset(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        api_key=args.api_key
    )

