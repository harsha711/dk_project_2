"""
Dataset utilities for Hugging Face teeth dataset integration
"""
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
from PIL import Image
import random


class TeethDatasetManager:
    """Manager for RayanAi/Main_teeth_dataset from Hugging Face"""

    def __init__(self, dataset_name: str = "RayanAi/Main_teeth_dataset"):
        self.dataset_name = dataset_name
        self.dataset = None
        self.current_index = 0

    def load_dataset(self) -> Dict:
        """
        Load the teeth dataset from Hugging Face

        Returns:
            dict with status and message
        """
        try:
            self.dataset = load_dataset(self.dataset_name)
            total_samples = len(self.dataset['train'])

            return {
                "success": True,
                "message": f"✅ Loaded {total_samples} dental X-ray samples",
                "total_samples": total_samples
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Failed to load dataset: {str(e)}",
                "total_samples": 0
            }

    def get_sample(self, index: int) -> Dict:
        """
        Get a specific sample from the dataset

        Args:
            index: Sample index

        Returns:
            dict with image, label, and metadata
        """
        if self.dataset is None:
            return {
                "success": False,
                "error": "Dataset not loaded. Call load_dataset() first."
            }

        try:
            sample = self.dataset['train'][index]

            return {
                "success": True,
                "image": sample['image'],
                "label": sample['label'],
                "index": index,
                "total": len(self.dataset['train'])
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get sample {index}: {str(e)}"
            }

    def get_random_sample(self) -> Dict:
        """Get a random sample from the dataset"""
        if self.dataset is None:
            return {
                "success": False,
                "error": "Dataset not loaded. Call load_dataset() first."
            }

        total_samples = len(self.dataset['train'])
        random_index = random.randint(0, total_samples - 1)

        return self.get_sample(random_index)

    def get_batch(self, start_idx: int, batch_size: int = 10) -> Dict:
        """
        Get a batch of samples

        Args:
            start_idx: Starting index
            batch_size: Number of samples to retrieve

        Returns:
            dict with list of samples
        """
        if self.dataset is None:
            return {
                "success": False,
                "error": "Dataset not loaded."
            }

        try:
            total_samples = len(self.dataset['train'])
            end_idx = min(start_idx + batch_size, total_samples)

            samples = []
            for i in range(start_idx, end_idx):
                sample = self.dataset['train'][i]
                samples.append({
                    "image": sample['image'],
                    "label": sample['label'],
                    "index": i
                })

            return {
                "success": True,
                "samples": samples,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "total": total_samples
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get batch: {str(e)}"
            }

    def get_samples_by_label(self, label: int, limit: int = 10) -> Dict:
        """
        Get samples filtered by label

        Args:
            label: Class label (0 or 1)
            limit: Maximum number of samples to return

        Returns:
            dict with filtered samples
        """
        if self.dataset is None:
            return {
                "success": False,
                "error": "Dataset not loaded."
            }

        try:
            filtered_samples = []
            count = 0

            for idx, sample in enumerate(self.dataset['train']):
                if sample['label'] == label and count < limit:
                    filtered_samples.append({
                        "image": sample['image'],
                        "label": sample['label'],
                        "index": idx
                    })
                    count += 1

                if count >= limit:
                    break

            return {
                "success": True,
                "samples": filtered_samples,
                "label": label,
                "count": len(filtered_samples)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to filter samples: {str(e)}"
            }

    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the dataset

        Returns:
            dict with dataset statistics
        """
        if self.dataset is None:
            return {
                "success": False,
                "error": "Dataset not loaded."
            }

        try:
            train_data = self.dataset['train']
            total_samples = len(train_data)

            # Count labels
            label_counts = {0: 0, 1: 0}
            for sample in train_data:
                label = sample['label']
                if label in label_counts:
                    label_counts[label] += 1

            return {
                "success": True,
                "total_samples": total_samples,
                "label_0_count": label_counts[0],
                "label_1_count": label_counts[1],
                "image_size": "512x512",
                "dataset_name": self.dataset_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get stats: {str(e)}"
            }

    def create_comparison_grid(self, samples: List[Dict], cols: int = 4) -> Image.Image:
        """
        Create a grid of sample images

        Args:
            samples: List of sample dicts with 'image' key
            cols: Number of columns in grid

        Returns:
            PIL Image with grid layout
        """
        if not samples:
            # Return blank image
            return Image.new('RGB', (512, 512), color='white')

        images = [s['image'] for s in samples]
        num_images = len(images)
        rows = (num_images + cols - 1) // cols

        # Assuming all images are 512x512
        img_width, img_height = 512, 512
        grid_width = cols * img_width
        grid_height = rows * img_height

        # Create blank grid
        grid = Image.new('RGB', (grid_width, grid_height), color='white')

        # Paste images
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * img_width
            y = row * img_height
            grid.paste(img, (x, y))

        return grid


def batch_analyze_samples(
    samples: List[Dict],
    model_choice: str,
    openai_client,
    groq_client
) -> List[Dict]:
    """
    Batch analyze multiple dental X-ray samples

    Args:
        samples: List of sample dicts with 'image' key
        model_choice: Vision model to use
        openai_client: OpenAI client
        groq_client: Groq client (unused for vision)

    Returns:
        List of analysis results
    """
    from api_utils import analyze_xray_gpt4v, analyze_xray_gemini
    from image_utils import parse_vision_response

    results = []

    for sample in samples:
        image = sample['image']

        # Analyze with selected model
        if model_choice == "GPT-4o Vision":
            result = analyze_xray_gpt4v(image, openai_client)
        else:
            result = analyze_xray_gemini(image)

        # Parse response
        if result["success"]:
            parsed = parse_vision_response(result["response"])
            results.append({
                "index": sample.get('index', -1),
                "success": True,
                "parsed": parsed,
                "raw": result["response"]
            })
        else:
            results.append({
                "index": sample.get('index', -1),
                "success": False,
                "error": result.get("error", "Unknown error")
            })

    return results


def export_analysis_results(results: List[Dict], filename: str = "analysis_results.json") -> bool:
    """
    Export batch analysis results to JSON file

    Args:
        results: List of analysis result dicts
        filename: Output filename

    Returns:
        True if successful
    """
    import json

    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False
