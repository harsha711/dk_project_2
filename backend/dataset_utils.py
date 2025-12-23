"""
Dataset utilities for dental X-ray dataset integration
SIMPLIFIED: Loads from local YOLO dataset if available, otherwise HuggingFace
"""
from datasets import load_dataset
from typing import Dict, Optional, List
from PIL import Image
from collections import OrderedDict
import numpy as np
import os
from pathlib import Path
import random
import io


class TeethDatasetManager:
    """
    Manager for dental X-ray dataset
    Tries local YOLO format dataset first, falls back to HuggingFace
    """

    def __init__(self, dataset_name: str = "RayanAi/Main_teeth_dataset", cache_size: int = 10):
        self.dataset_name = dataset_name
        self.dataset = None
        self.total_samples = 0
        self.loaded = False
        self.cache_size = cache_size
        self.image_cache = OrderedDict()
        
        # Local dataset paths to check
        self.local_dataset_paths = [
            "Dental-X-ray-1/train/images",
            "datasets/Dental-X-ray-1/train/images",
            "../Dental-X-ray-1/train/images",
            "Dental-Xray-1/train/images",
            "datasets/Dental-Xray-1/train/images",
        ]
        
        self.use_local = False
        self.local_image_paths = []
        self.local_dataset_dir = None

    def _find_local_dataset(self) -> Optional[str]:
        """Find local YOLO format dataset"""
        for path in self.local_dataset_paths:
            full_path = os.path.abspath(path)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                # Check if it has images
                image_files = list(Path(full_path).glob("*.jpg")) + list(Path(full_path).glob("*.png"))
                if len(image_files) > 0:
                    print(f"[LOCAL] Found local dataset at: {full_path}")
                    return full_path
        
        # Also check parent directory
        current_dir = os.getcwd()
        parent_dirs = [
            os.path.join(current_dir, "..", "Dental-X-ray-1", "train", "images"),
            os.path.join(current_dir, "Dental-X-ray-1", "train", "images"),
        ]
        for path in parent_dirs:
            full_path = os.path.abspath(path)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                image_files = list(Path(full_path).glob("*.jpg")) + list(Path(full_path).glob("*.png"))
                if len(image_files) > 0:
                    print(f"[LOCAL] Found local dataset at: {full_path}")
                    return full_path
        
        return None

    def load_metadata(self) -> Dict:
        """Load dataset metadata"""
        try:
            # Try local dataset first
            local_path = self._find_local_dataset()
            if local_path:
                self.local_dataset_dir = local_path
                # Get all image files
                self.local_image_paths = sorted(
                    list(Path(local_path).glob("*.jpg")) + 
                    list(Path(local_path).glob("*.png")) +
                    list(Path(local_path).glob("*.jpeg"))
                )
                self.total_samples = len(self.local_image_paths)
                self.use_local = True
                self.loaded = True
                print(f"[LOCAL] Loaded {self.total_samples} images from local dataset")
                return {
                    "success": True,
                    "message": f"✅ Local dataset ready: {self.total_samples} samples",
                    "total_samples": self.total_samples,
                    "source": "local"
                }
            
            # Fallback to HuggingFace
            print("[HUGGINGFACE] Local dataset not found, using HuggingFace...")
            self.use_local = False
            self.total_samples = 1206  # Known HuggingFace size
            self.loaded = True
            return {
                "success": True,
                "message": f"✅ HuggingFace dataset ready: {self.total_samples} samples",
                "total_samples": self.total_samples,
                "source": "huggingface"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Failed: {str(e)}",
                "total_samples": 0
            }

    def _ensure_dataset_loaded(self):
        """Load HuggingFace dataset if not already loaded"""
        if self.dataset is None and not self.use_local:
            print("[HUGGINGFACE] Loading from HuggingFace...")
            self.dataset = load_dataset(self.dataset_name, split="train")
            print(f"[HUGGINGFACE] Loaded {len(self.dataset)} samples")

    def get_sample(self, index: int) -> Dict:
        """Get a single sample - from local or HuggingFace
        Filters out binary/mask images and only returns real X-rays
        """
        
        max_attempts = 50  # Try up to 50 different indices to find a valid X-ray
        original_index = index
        
        # Check cache first (but validate cached images too)
        if index in self.image_cache:
            cached = self.image_cache[index]
            # Validate cached image is not binary
            if cached.get("image"):
                arr = np.array(cached["image"])
                unique_vals = len(np.unique(arr))
                if unique_vals >= 10:  # Valid X-ray
                    print(f"[CACHE HIT] Sample {index} (valid)")
                    return cached
                else:
                    # Cached image is binary, remove from cache
                    print(f"[CACHE] Removing binary cached image {index}")
                    self.image_cache.pop(index, None)

        try:
            if self.use_local:
                # Load from local filesystem
                if not self.local_image_paths:
                    return {"success": False, "error": "No local images found", "image": None, "index": index}
                
                # Clamp index
                index = index % len(self.local_image_paths)
                image_path = self.local_image_paths[index]
                
                # Load image
                image = Image.open(image_path)
                image.load()  # Force load
                
                # CRITICAL: Clean image conversion for Gradio display
                buffer = io.BytesIO()
                
                # Ensure RGB mode for display - FIXED: Proper grayscale to RGB conversion
                if image.mode == 'L':
                    # Grayscale to RGB - preserve full range, don't threshold
                    arr = np.array(image)  # Shape: (H, W)
                    if len(arr.shape) == 2:
                        # Stack to create RGB: (H, W, 3)
                        arr_rgb = np.stack([arr, arr, arr], axis=-1)
                        image = Image.fromarray(arr_rgb.astype(np.uint8), mode='RGB')
                    else:
                        image = image.convert('RGB')
                elif image.mode == 'RGBA':
                    # Remove alpha channel
                    image = image.convert('RGB')
                elif image.mode != 'RGB':
                    # Any other mode - convert
                    image = image.convert('RGB')
                
                # Save and reload to ensure clean format (fixes corruption issues)
                image.save(buffer, format='PNG')
                buffer.seek(0)
                image = Image.open(buffer)
                image.load()  # Force load to ensure data is in memory
                buffer.close()
                
                # Create result
                result = {
                    "success": True,
                    "image": image,
                    "label": 0,  # Local dataset doesn't have labels in this format
                    "index": index,
                    "total": len(self.local_image_paths)
                }
                
                # Cache it
                self._add_to_cache(index, result)
                
                print(f"[LOCAL] Loaded sample {index}: {image.size}, mode={image.mode}, file={image_path.name}")
                return result
            
            else:
                # Load from HuggingFace with filtering for real X-rays
                self._ensure_dataset_loaded()
                
                # Try multiple indices to find a valid X-ray (skip binary/mask images)
                for attempt in range(max_attempts):
                    try_index = (original_index + attempt) % self.total_samples
                    
                    # Get sample
                    sample = self.dataset[try_index]
                    
                    # Get image - simple extraction
                    image = sample.get('image')
                    
                    if image is None:
                        continue
                    
                    # Convert to PIL if needed
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                    
                    # Convert to array for validation
                    arr = np.array(image)
                    
                    # FILTER: Skip binary/mask images
                    unique_values = len(np.unique(arr))
                    mean_value = arr.mean()
                    
                    print(f"[CHECK] Sample {try_index}: unique={unique_values}, mean={mean_value:.1f}")
                    
                    # Real X-rays have:
                    # - Many unique values (10+)
                    # - Reasonable mean brightness (5-250)
                    if unique_values < 10:
                        print(f"[SKIP] Sample {try_index} is binary/mask (only {unique_values} unique values)")
                        continue
                    
                    if mean_value < 5 or mean_value > 250:
                        print(f"[SKIP] Sample {try_index} has bad brightness (mean={mean_value:.1f})")
                        continue
                    
                    # Valid X-ray found!
                    print(f"[VALID] Sample {try_index}: unique={unique_values}, mean={mean_value:.1f}")
                    
                    # CRITICAL: Clean image conversion for Gradio display
                    # Save to buffer and reload to ensure clean format
                    buffer = io.BytesIO()
                    
                    # Ensure RGB mode for display - FIXED: Proper grayscale to RGB conversion
                    if image.mode == 'L':
                        # Grayscale to RGB - preserve full range, don't threshold
                        # Method: Convert each grayscale pixel to RGB by duplicating the value
                        if len(arr.shape) == 2:
                            # Stack to create RGB: (H, W, 3)
                            arr_rgb = np.stack([arr, arr, arr], axis=-1)
                            image = Image.fromarray(arr_rgb.astype(np.uint8), mode='RGB')
                        else:
                            # Fallback to PIL convert
                            image = image.convert('RGB')
                    elif image.mode == 'RGBA':
                        # Remove alpha channel
                        image = image.convert('RGB')
                    elif image.mode != 'RGB':
                        # Any other mode - convert
                        image = image.convert('RGB')
                    
                    # Save and reload to ensure clean format (fixes corruption issues)
                    image.save(buffer, format='PNG')
                    buffer.seek(0)
                    image = Image.open(buffer)
                    image.load()  # Force load to ensure data is in memory
                    buffer.close()
                    
                    # Create result
                    result = {
                        "success": True,
                        "image": image,
                        "label": sample.get('label', 0),
                        "index": try_index,
                        "total": self.total_samples,
                        "skipped": attempt  # How many we skipped to find this valid image
                    }
                    
                    # Cache it
                    self._add_to_cache(try_index, result)
                    
                    print(f"[HUGGINGFACE] Loaded valid sample {try_index}: {image.size}, mode={image.mode}")
                    return result
                
                # All attempts failed - couldn't find a valid X-ray
                return {
                    "success": False,
                    "error": f"Could not find valid X-ray after {max_attempts} attempts. Dataset may contain mostly binary masks.",
                    "image": None,
                    "index": original_index,
                    "total": self.total_samples
                }

        except Exception as e:
            print(f"[ERROR] Failed to load sample {index}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "image": None,
                "index": index,
                "total": self.total_samples if self.total_samples > 0 else 0
            }

    def _add_to_cache(self, index: int, data: Dict):
        """Add to LRU cache"""
        if index in self.image_cache:
            self.image_cache.move_to_end(index)
            return
        
        self.image_cache[index] = data
        
        if len(self.image_cache) > self.cache_size:
            self.image_cache.pop(next(iter(self.image_cache)))

    def get_random_sample(self) -> Dict:
        """Get random sample"""
        if self.use_local and self.local_image_paths:
            index = random.randint(0, len(self.local_image_paths) - 1)
        else:
            index = random.randint(0, self.total_samples - 1)
        return self.get_sample(index)

    def get_next_sample(self, current_index: int) -> Dict:
        """Get next sample"""
        if self.use_local and self.local_image_paths:
            return self.get_sample((current_index + 1) % len(self.local_image_paths))
        return self.get_sample((current_index + 1) % self.total_samples)

    def get_previous_sample(self, current_index: int) -> Dict:
        """Get previous sample"""
        if self.use_local and self.local_image_paths:
            return self.get_sample((current_index - 1) % len(self.local_image_paths))
        return self.get_sample((current_index - 1) % self.total_samples)

    def clear_cache(self):
        """Clear cache"""
        self.image_cache.clear()
        return "✅ Cache cleared"

    def get_dataset_stats(self) -> Dict:
        """Get dataset stats"""
        return {
            "success": True,
            "total_samples": self.total_samples,
            "dataset_name": self.dataset_name if not self.use_local else "Local YOLO Dataset",
            "cache_count": len(self.image_cache),
            "source": "local" if self.use_local else "huggingface"
        }

    @property
    def is_loaded(self):
        return self.loaded

    def scan_dataset_quality(self, num_samples: int = 100) -> Dict:
        """
        Scan dataset to check quality - count valid X-rays vs binary/mask images
        
        Args:
            num_samples: Number of samples to check (default: 100)
        
        Returns:
            Dict with quality statistics
        """
        if not self.loaded:
            self.load_metadata()
        
        if self.use_local:
            return {
                "success": False,
                "message": "Quality scan only available for HuggingFace dataset"
            }
        
        self._ensure_dataset_loaded()
        
        valid = 0
        binary = 0
        too_dark = 0
        too_bright = 0
        
        samples_to_check = min(num_samples, self.total_samples)
        
        print(f"\n[QUALITY SCAN] Checking {samples_to_check} samples...")
        
        for i in range(samples_to_check):
            try:
                sample = self.dataset[i]
                image = sample.get('image')
                
                if image is None:
                    continue
                
                if isinstance(image, np.ndarray):
                    arr = image
                else:
                    arr = np.array(image)
                
                unique_vals = len(np.unique(arr))
                mean_val = arr.mean()
                
                if unique_vals < 10:
                    binary += 1
                elif mean_val < 5:
                    too_dark += 1
                elif mean_val > 250:
                    too_bright += 1
                else:
                    valid += 1
                    
            except Exception as e:
                print(f"[ERROR] Failed to check sample {i}: {e}")
                continue
        
        total_checked = valid + binary + too_dark + too_bright
        valid_percent = (valid / total_checked * 100) if total_checked > 0 else 0
        
        result = {
            "success": True,
            "total_checked": total_checked,
            "valid_xrays": valid,
            "binary_masks": binary,
            "too_dark": too_dark,
            "too_bright": too_bright,
            "valid_percent": valid_percent,
            "message": f"Dataset quality: {valid} valid X-rays ({valid_percent:.1f}%), {binary} binary/masks, {too_dark} too dark, {too_bright} too bright"
        }
        
        print(f"\n[QUALITY SCAN RESULTS]")
        print(f"  Valid X-rays: {valid} ({valid_percent:.1f}%)")
        print(f"  Binary/masks: {binary}")
        print(f"  Too dark: {too_dark}")
        print(f"  Too bright: {too_bright}")
        print(f"  Total checked: {total_checked}")
        
        return result


# Singleton instance
_manager_instance = None

def get_teeth_dataset_manager() -> TeethDatasetManager:
    """Get or create singleton manager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = TeethDatasetManager()
    return _manager_instance
