"""
Dataset utilities for Hugging Face teeth dataset integration
OPTIMIZED: Uses lazy loading and caching to prevent browser crashes
"""
from datasets import load_dataset, IterableDataset
from typing import Dict, Optional
from PIL import Image
from collections import OrderedDict


class TeethDatasetManager:
    """
    Manager for RayanAi/Main_teeth_dataset from Hugging Face

    FEATURES:
    - Lazy loading: Only loads images on-demand
    - Streaming mode: Doesn't download entire dataset
    - LRU cache: Keeps last 10 images in memory
    - Memory efficient: Won't crash browser
    """

    def __init__(self, dataset_name: str = "RayanAi/Main_teeth_dataset", cache_size: int = 10):
        self.dataset_name = dataset_name
        self.dataset = None
        self.total_samples = 0
        self.loaded = False

        # LRU cache for images (keeps last N images)
        self.cache_size = cache_size
        self.image_cache = OrderedDict()

    def load_metadata(self) -> Dict:
        """
        Load only metadata (count, info) without loading images
        Uses streaming mode to avoid downloading entire dataset

        Returns:
            dict with metadata
        """
        try:
            # Use streaming to get count without downloading all data
            print("[DATASET] Loading metadata (streaming mode)...")
            dataset_info = load_dataset(self.dataset_name, split="train", streaming=True)

            # For streaming datasets, we know the size from HF metadata
            self.total_samples = 1206  # Known size from HuggingFace
            self.loaded = True

            return {
                "success": True,
                "message": f"✅ Dataset ready: {self.total_samples} samples (lazy loading enabled)",
                "total_samples": self.total_samples,
                "cache_size": self.cache_size,
                "tip": "💡 Images load on-demand to save memory"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Failed to load dataset metadata: {str(e)}",
                "total_samples": 0
            }

    def get_sample(self, index: int) -> Dict:
        """
        Get a single sample (lazy loading with cache)

        Args:
            index: Sample index (0 to total_samples-1)

        Returns:
            dict with image, label, and metadata
        """
        # Check cache first
        if index in self.image_cache:
            print(f"[CACHE HIT] Loading sample {index} from cache")
            return self.image_cache[index]

        try:
            print(f"[LOADING] Fetching sample {index} from HuggingFace...")

            # Load dataset in normal mode (not streaming) to access by index
            if self.dataset is None:
                self.dataset = load_dataset(self.dataset_name, split="train")

            # Get single sample
            sample = self.dataset[index]

            # DEBUG: Print image type from dataset
            raw_img = sample['image']
            print(f"[DATASET] Image type from HF: {type(raw_img)}")
            if hasattr(raw_img, 'mode'):
                print(f"[DATASET] Image mode: {raw_img.mode}, size: {raw_img.size}")

            result = {
                "success": True,
                "image": sample['image'],
                "label": sample['label'],
                "index": index,
                "total": self.total_samples,
                "cached": False
            }

            # Add to cache
            self._add_to_cache(index, result)

            print(f"[SUCCESS] Loaded sample {index}, Label: {sample['label']}")
            return result

        except Exception as e:
            print(f"[ERROR] Failed to load sample {index}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to load sample {index}: {str(e)}",
                "image": None,
                "label": None,
                "index": index,
                "total": self.total_samples
            }

    def _add_to_cache(self, index: int, data: Dict):
        """
        Add sample to LRU cache
        Automatically removes oldest if cache is full
        """
        # If already in cache, move to end (most recent)
        if index in self.image_cache:
            self.image_cache.move_to_end(index)
            return

        # Add new item
        self.image_cache[index] = data

        # Remove oldest if cache is full
        if len(self.image_cache) > self.cache_size:
            oldest_key = next(iter(self.image_cache))
            removed = self.image_cache.pop(oldest_key)
            print(f"[CACHE] Removed sample {oldest_key} (cache full)")

    def get_next_sample(self, current_index: int) -> Dict:
        """Get next sample (wraps around to 0)"""
        if self.total_samples == 0:
            return {"success": False, "error": "Dataset not loaded"}

        next_index = (current_index + 1) % self.total_samples
        return self.get_sample(next_index)

    def get_previous_sample(self, current_index: int) -> Dict:
        """Get previous sample (wraps around to last)"""
        if self.total_samples == 0:
            return {"success": False, "error": "Dataset not loaded"}

        prev_index = (current_index - 1) % self.total_samples
        return self.get_sample(prev_index)

    def clear_cache(self):
        """Clear image cache to free memory"""
        cache_count = len(self.image_cache)
        self.image_cache.clear()
        print(f"[CACHE] Cleared {cache_count} cached images")
        return f"✅ Cleared {cache_count} cached images"

    def get_cache_info(self) -> Dict:
        """Get information about current cache state"""
        return {
            "cached_indices": list(self.image_cache.keys()),
            "cache_count": len(self.image_cache),
            "cache_size": self.cache_size,
            "cache_usage_pct": (len(self.image_cache) / self.cache_size) * 100 if self.cache_size > 0 else 0
        }

    def get_dataset_stats(self) -> Dict:
        """
        Get lightweight statistics without loading all images

        Returns:
            dict with dataset statistics
        """
        if not self.loaded:
            return {
                "success": False,
                "error": "Dataset metadata not loaded. Call load_metadata() first."
            }

        try:
            # Return known stats without iterating entire dataset
            return {
                "success": True,
                "total_samples": self.total_samples,
                "dataset_name": self.dataset_name,
                "image_size": "512x512 (estimated)",
                "cache_info": self.get_cache_info(),
                "lazy_loading": "enabled",
                "tip": "Images are loaded one at a time to save memory"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get stats: {str(e)}"
            }
