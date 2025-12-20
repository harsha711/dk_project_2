# 📊 Dataset Features Documentation

## Overview

The enhanced Dental AI Platform now includes **Tab 3: Dataset Explorer** which integrates the **RayanAi/Main_teeth_dataset** from Hugging Face (1,206 dental X-ray samples).

## Dataset Information

- **Source:** https://huggingface.co/datasets/RayanAi/Main_teeth_dataset
- **Total Samples:** 1,206 dental X-ray images
- **Image Size:** 512x512 pixels
- **Format:** JPEG images
- **Labels:** Binary classification (0: images, 1: labels)
- **Split:** Training set only

## New Features in Enhanced App

### 1. Dataset Loading
```python
# Automatically downloads and caches dataset from Hugging Face
dataset_manager.load_dataset()
```

**Features:**
- One-click dataset loading
- Automatic caching (downloads once, reuses)
- Dataset statistics display
- Label distribution

### 2. Sample Navigation

**Browse Methods:**
- ⬅️ **Previous/Next** - Navigate sequentially through samples
- 🎲 **Random Sample** - Jump to random X-ray
- **Jump to Index** - Go directly to specific sample number

**Display Info:**
- Sample index (0-1205)
- Label (0 or 1)
- Total samples count

### 3. Batch Processing

**What it does:**
Process multiple X-rays from the dataset using AI vision models in one click.

**Parameters:**
- **Start Index:** Where to begin in dataset (0-1205)
- **Batch Size:** How many samples to process (1-50)
- **Analysis Model:** GPT-4o Vision or Gemini Vision

**Output:**
- Number of samples processed
- Success/failure for each sample
- Teeth count for successful analyses
- Overall success rate

**Example Use:**
```
Start Index: 0
Batch Size: 10
Model: Gemini Vision

Result: Processes samples #0-9, shows teeth detection for each
```

## Architecture

### New Module: `dataset_utils.py`

**Classes:**
- `TeethDatasetManager` - Main dataset interface

**Key Methods:**

```python
# Load dataset
load_dataset() → dict

# Get single sample
get_sample(index: int) → dict

# Get random sample
get_random_sample() → dict

# Get batch of samples
get_batch(start_idx: int, batch_size: int) → dict

# Filter by label
get_samples_by_label(label: int, limit: int) → dict

# Get statistics
get_dataset_stats() → dict

# Create image grid
create_comparison_grid(samples: list, cols: int) → Image

# Batch analysis
batch_analyze_samples(samples, model, clients) → list
```

## Usage Guide

### Step 1: Load Dataset

1. Click **Tab 3: Dataset Explorer**
2. Click **"📥 Load Dataset from Hugging Face"**
3. Wait 10-30 seconds (first time only)
4. See dataset statistics

**Output:**
```
✅ Dataset Loaded Successfully

Total Samples: 1,206
Label 0 (images): XXX
Label 1 (labels): XXX
Image Size: 512x512
Dataset: RayanAi/Main_teeth_dataset
```

### Step 2: Browse Samples

**Option A: Sequential Navigation**
1. Click **"Next ➡️"** to go forward
2. Click **"⬅️ Previous"** to go backward
3. View sample image and metadata

**Option B: Random Exploration**
1. Click **"🎲 Random Sample"**
2. Get random X-ray from dataset

**Option C: Jump to Specific Sample**
1. Enter index number (0-1205) in "Jump to Index"
2. Click **"Go to Index"**
3. View that specific sample

### Step 3: Batch Analysis

**Use Case:** Analyze multiple X-rays automatically

**Steps:**
1. Set **Start Index** (e.g., 0)
2. Set **Batch Size** (e.g., 5-10)
3. Select **Analysis Model** (Gemini Vision recommended for speed)
4. Click **"⚡ Run Batch Analysis"**
5. Wait for progress bar
6. View results

**Example Output:**
```
Batch Analysis Complete

Samples Processed: 10
Model Used: Gemini Vision

---

Sample #0: 2 wisdom teeth detected
Sample #1: 0 wisdom teeth detected
Sample #2: 4 wisdom teeth detected
Sample #3: 1 wisdom teeth detected
Sample #4: ❌ API error
...

Success Rate: 9/10
```

## Performance Considerations

### Dataset Loading
- **First Time:** 10-30 seconds (downloads ~90 MB)
- **Subsequent:** Instant (uses cache)
- **Cache Location:** `~/.cache/huggingface/datasets/`

### Batch Processing Time

| Batch Size | GPT-4o Vision | Gemini Vision |
|------------|---------------|---------------|
| 5 samples  | ~60s          | ~30s          |
| 10 samples | ~120s         | ~60s          |
| 20 samples | ~240s         | ~120s         |
| 50 samples | ~600s         | ~300s         |

**Recommendation:** Use Gemini Vision for batch processing (2x faster, cheaper)

### Cost Estimation

**Per 100 Samples:**
- **GPT-4o Vision:** ~$50-100 (depending on response length)
- **Gemini Vision:** ~$1-2 (generous free tier)

**Tip:** Start with small batches (5-10) to test, then scale up.

## Code Examples

### Manual Usage in Python

```python
from dataset_utils import TeethDatasetManager

# Initialize
manager = TeethDatasetManager()

# Load dataset
result = manager.load_dataset()
print(result['message'])

# Get sample
sample = manager.get_sample(42)
image = sample['image']  # PIL Image
label = sample['label']  # 0 or 1

# Get random sample
random_sample = manager.get_random_sample()

# Get batch
batch = manager.get_batch(start_idx=0, batch_size=10)
for sample in batch['samples']:
    print(f"Sample {sample['index']}: Label {sample['label']}")

# Get statistics
stats = manager.get_dataset_stats()
print(f"Total: {stats['total_samples']}")
print(f"Label 0: {stats['label_0_count']}")
print(f"Label 1: {stats['label_1_count']}")

# Filter by label
label_0_samples = manager.get_samples_by_label(label=0, limit=5)
```

### Batch Analysis Example

```python
from dataset_utils import batch_analyze_samples
from api_utils import init_clients

# Initialize
manager = TeethDatasetManager()
manager.load_dataset()
openai_client, groq_client = init_clients()

# Get batch
batch = manager.get_batch(0, 5)
samples = batch['samples']

# Analyze all samples
results = batch_analyze_samples(
    samples=samples,
    model_choice="Gemini Vision",
    openai_client=openai_client,
    groq_client=groq_client
)

# Process results
for result in results:
    if result['success']:
        teeth_found = result['parsed'].get('teeth_found', [])
        print(f"Sample {result['index']}: {len(teeth_found)} teeth")
    else:
        print(f"Sample {result['index']}: Error - {result['error']}")
```

## Advanced Features

### Export Analysis Results

```python
from dataset_utils import export_analysis_results

# After batch analysis
export_analysis_results(results, "batch_results.json")

# Creates JSON file with:
# - Sample indices
# - Detection results
# - Bounding boxes
# - Summaries
```

### Create Image Grid

```python
manager = TeethDatasetManager()
manager.load_dataset()

# Get samples
batch = manager.get_batch(0, 16)

# Create 4x4 grid
grid_image = manager.create_comparison_grid(
    samples=batch['samples'],
    cols=4
)

# Save or display
grid_image.save("grid.png")
```

## Troubleshooting

### Dataset Won't Load

**Error:** "Failed to load dataset"

**Solutions:**
1. Check internet connection
2. Verify Hugging Face is accessible
3. Clear cache: `rm -rf ~/.cache/huggingface/datasets/`
4. Try again

### Batch Processing Slow

**Problem:** Taking too long

**Solutions:**
1. Reduce batch size (try 5 instead of 20)
2. Use Gemini Vision instead of GPT-4o
3. Run during off-peak hours
4. Check API rate limits

### Out of Memory

**Error:** "Memory error" during batch processing

**Solutions:**
1. Reduce batch size
2. Process in smaller chunks
3. Close other applications
4. Use cloud instance with more RAM

## Integration with Existing Tabs

### Tab 1 → Tab 3 Workflow

1. **Browse dataset** in Tab 3
2. Find interesting sample
3. **Copy image** (right-click, copy)
4. **Switch to Tab 1**
5. Upload copied image for detailed analysis

### Tab 3 → Tab 2 Workflow

1. **Run batch analysis** in Tab 3
2. Note patterns/findings
3. **Switch to Tab 2**
4. Ask questions about findings across models

## Future Enhancements

Potential additions to dataset features:

- [ ] **Export to CSV** - Save batch results as spreadsheet
- [ ] **Filter by teeth count** - Show only samples with N teeth
- [ ] **Annotation editor** - Manually correct bounding boxes
- [ ] **Model fine-tuning** - Train custom detector on dataset
- [ ] **Comparison view** - Side-by-side samples
- [ ] **Search by similarity** - Find similar X-rays
- [ ] **Data augmentation** - Generate variations for training
- [ ] **Quality scoring** - Rate image quality automatically

## Dataset Statistics

After loading, you can view:

```
Total Samples: 1,206
Label Distribution:
  - Label 0 (images): ~XXX samples
  - Label 1 (labels): ~XXX samples

Image Specifications:
  - Format: JPEG
  - Dimensions: 512x512 pixels
  - Color Mode: RGB
  - File Size: ~50-100 KB per image

Storage:
  - Parquet format: 64.4 MB
  - Full dataset: 90.2 MB
  - Cached locally after first download
```

## API Usage Tracking

When using batch processing, monitor your usage:

**Gemini Vision (Free Tier):**
- 15 requests/minute
- 1,500 requests/day
- Safe for ~500 samples/day

**GPT-4o Vision (Paid):**
- 500 requests/minute (tier 1)
- 10,000 requests/day (tier 1)
- Cost: ~$0.50 per sample

**Recommendation:** Use Gemini for exploration, GPT-4o for critical analysis.

---

## Quick Commands

```bash
# Install new dependencies
pip install -r requirements.txt

# Run enhanced app
python dental_ai_enhanced.py

# Test dataset loading
python -c "from dataset_utils import TeethDatasetManager; m = TeethDatasetManager(); print(m.load_dataset())"
```

---

**Tab 3 gives you:** Production-ready dataset exploration and batch processing capabilities for dental AI research and development.
