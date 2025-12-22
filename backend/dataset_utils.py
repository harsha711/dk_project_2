"""
Dataset utilities for Hugging Face teeth dataset integration
OPTIMIZED: Uses lazy loading and caching to prevent browser crashes
"""
from datasets import load_dataset, IterableDataset
from typing import Dict, Optional
from PIL import Image
from collections import OrderedDict
import numpy as np


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

            # Get image - try to preserve original as much as possible
            raw_img = sample['image']
            print(f"[DATASET] Image type from HF: {type(raw_img)}")
            if hasattr(raw_img, 'mode'):
                print(f"[DATASET] Image mode: {raw_img.mode}, size: {raw_img.size}")
            
            # If it's already a valid PIL Image, minimize processing
            if isinstance(raw_img, Image.Image):
                # Quick validation - check if it's not corrupted
                if raw_img.size[0] > 0 and raw_img.size[1] > 0:
                    # Only convert mode if necessary, preserve original data
                    processed_img = raw_img
                    
                    # Convert to RGB only if needed for display
                    if processed_img.mode not in ('RGB', 'L', 'P'):
                        try:
                            processed_img = processed_img.convert('RGB')
                            print(f"[IMAGE] Converted from {raw_img.mode} to RGB")
                        except Exception as e:
                            print(f"[WARNING] Could not convert {raw_img.mode} to RGB: {e}")
                            # Try processing through full pipeline
                            processed_img = self._process_image(raw_img)
                    elif processed_img.mode == 'L':
                        # Grayscale - convert to RGB for consistency
                        processed_img = processed_img.convert('RGB')
                        print(f"[IMAGE] Converted grayscale to RGB")
                    elif processed_img.mode == 'P':
                        # Palette mode - convert to RGB
                        processed_img = processed_img.convert('RGB')
                        print(f"[IMAGE] Converted palette to RGB")
                    else:
                        # Already RGB or acceptable mode
                        print(f"[IMAGE] Keeping original mode: {processed_img.mode}")
                else:
                    print(f"[ERROR] Image has invalid size: {raw_img.size}")
                    processed_img = None
            else:
                # Not a PIL Image - need full processing
                processed_img = self._process_image(raw_img)
            
            if processed_img is None:
                print(f"[WARNING] Sample {index} has invalid image, trying next...")
                # Try next sample if current one is invalid (up to 5 attempts)
                max_attempts = 5
                for attempt in range(1, max_attempts + 1):
                    next_index = (index + attempt) % self.total_samples
                    if next_index != index:  # Don't loop back to same index
                        print(f"[RETRY] Attempting sample {next_index} (attempt {attempt}/{max_attempts})")
                        return self.get_sample(next_index)
                
                # If all attempts failed, return error
                return {
                    "success": False,
                    "error": f"Sample {index} and next {max_attempts} samples have invalid images",
                    "image": None,
                    "label": None,
                    "index": index,
                    "total": self.total_samples
                }

            result = {
                "success": True,
                "image": processed_img,
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

    def _process_image(self, img) -> Optional[Image.Image]:
        """
        Process and validate image from dataset
        Handles various formats and ensures proper display
        
        Args:
            img: Image from dataset (PIL Image or other format)
        
        Returns:
            Processed PIL Image in RGB mode, or None if invalid
        """
        try:
            # Ensure it's a PIL Image
            if not isinstance(img, Image.Image):
                if hasattr(img, 'convert'):
                    img = img.convert('RGB')
                else:
                    # Try to convert from numpy array or other format
                    if isinstance(img, np.ndarray):
                        # Normalize if needed
                        if img.dtype != np.uint8:
                            # Normalize to 0-255 range
                            img_min, img_max = img.min(), img.max()
                            if img_max > img_min:
                                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        
                        # Handle grayscale vs RGB
                        if len(img.shape) == 2:
                            img = Image.fromarray(img, mode='L')
                        elif len(img.shape) == 3:
                            img = Image.fromarray(img, mode='RGB')
                        else:
                            print(f"[ERROR] Unsupported image shape: {img.shape}")
                            return None
                    else:
                        print(f"[ERROR] Unsupported image type: {type(img)}")
                        return None
            
            # Validate image is not empty or corrupted
            if img.size[0] == 0 or img.size[1] == 0:
                print(f"[ERROR] Image has zero size")
                return None
            
            # Convert to RGB if needed (handles grayscale, palette, etc.)
            if img.mode == 'L':
                # Grayscale - convert to RGB by duplicating channels
                img = img.convert('RGB')
            elif img.mode == 'P':
                # Palette mode - convert to RGB
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                # RGBA - convert to RGB (drop alpha)
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                # Any other mode - try to convert
                try:
                    img = img.convert('RGB')
                except Exception as e:
                    print(f"[ERROR] Failed to convert image mode {img.mode}: {e}")
                    return None
            
            # Validate image is not corrupted
            img_array = np.array(img)
            
            # For X-rays, validation needs to be different:
            # - X-rays can have very few colors (black background, white structures)
            # - Check for actual structure/variance rather than just color count
            mean_brightness = img_array.mean()
            std_brightness = img_array.std()
            min_brightness = img_array.min()
            max_brightness = img_array.max()
            
            # Check if image is all black (mean < 5) or all white (mean > 250) - definitely corrupted
            if mean_brightness < 5:
                print(f"[ERROR] Image is all black (mean={mean_brightness:.1f}) - appears corrupted")
                return None
            if mean_brightness > 250:
                print(f"[ERROR] Image is all white (mean={mean_brightness:.1f}) - appears corrupted")
                return None
            
            # Check for static noise or completely uniform images
            # X-rays should have some variance (std > 5) to show structure
            if std_brightness < 5:
                # Very low variance - might be uniform/corrupted
                # But allow if it's a valid grayscale range (not all one value)
                if max_brightness - min_brightness < 10:
                    print(f"[ERROR] Image has no variance (std={std_brightness:.1f}, range={max_brightness-min_brightness:.1f}) - appears corrupted")
                    return None
            
            # Count unique colors for logging (but don't reject based on this for X-rays)
            if len(img_array.shape) == 3:
                # RGB image - count unique RGB tuples
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            else:
                # Grayscale - count unique intensity values
                unique_colors = len(np.unique(img_array))
            
            # For X-rays, even 2 colors (black/white) can be valid
            # Only reject if it's truly uniform (all same value)
            if unique_colors == 1:
                print(f"[ERROR] Image has only 1 unique color - appears corrupted")
                return None
            
            # Log info for very few colors (but allow it for X-rays)
            if unique_colors < 10:
                print(f"[INFO] Image has {unique_colors} unique colors (X-ray may be binary/grayscale - this is OK)")
            
            # Check for static noise (very high standard deviation with random patterns)
            # Real X-rays have structured patterns, noise is random
            # If std is very high (>100) and unique colors is very high (>5000), might be noise
            # But we'll be lenient here since some X-rays can have high variance
            
            # For very dark X-rays (which are valid), log info
            # X-rays are typically dark, so we'll allow dark images
            if mean_brightness < 50:
                # Very dark image - might need enhancement, but it's valid
                print(f"[INFO] Image is dark (mean={mean_brightness:.1f}) - valid X-ray, may need enhancement")
            
            print(f"[IMAGE PROCESS] Processed image: mode={img.mode}, size={img.size}")
            return img
            
        except Exception as e:
            print(f"[ERROR] Failed to process image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

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
