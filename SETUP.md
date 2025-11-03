# Setup Guide

This guide will help you set up and run the "Do We Date Alike?" analysis.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for version control)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/do-we-date-alike.git
cd do-we-date-alike
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If you encounter issues with tensorflow on M1/M2 Macs:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

### 4. Verify Installation

```bash
python -c "import cv2, tensorflow, numpy, scipy, matplotlib; print('All packages installed successfully!')"
```

## Data Preparation

### Option 1: Use Your Own Data

1. Collect couple images where both faces are clearly visible
2. Place images in the `data/raw/` directory following this structure:

```
data/raw/
├── couple_0001/
│   ├── person1.jpg
│   └── person2.jpg
├── couple_0002/
│   ├── person1.jpg
│   └── person2.jpg
└── ...
```

3. Create a metadata file:

```python
from src.data_collection import CoupleDataCollector

collector = CoupleDataCollector()
collector.add_couple_from_files(
    "path/to/person1.jpg",
    "path/to/person2.jpg",
    metadata={"source": "description", "names": "optional"}
)
```

### Option 2: Use Celebrity Couple Datasets

You can collect celebrity couple images from:
- Google Images (with permission)
- Public datasets like LFW (Labeled Faces in the Wild)
- Dating app research datasets (requires permission)

**Important**: Ensure you have rights to use any images and cite sources appropriately.

## Running the Analysis

### Quick Start (Complete Pipeline)

```bash
python run_pipeline.py
```

This will:
1. Extract facial embeddings from all couples
2. Perform statistical analysis
3. Generate visualizations
4. Save results to `results/` directory

### Step-by-Step Execution

#### Step 1: Extract Embeddings

```bash
python src/facial_analysis/extract_embeddings.py \
    --data-dir data/raw \
    --output-dir data/processed \
    --detector opencv \
    --model facenet
```

Options:
- `--detector`: opencv (fast), mtcnn (accurate), dlib (robust)
- `--model`: facenet (recommended), vggface, arcface

#### Step 2: Run Statistical Analysis

```bash
python src/statistical_analysis/analyze_similarity.py \
    --embeddings-file data/processed/embeddings_results.json \
    --output-file results/statistical_analysis.json
```

#### Step 3: Generate Visualizations

```bash
python src/visualization/create_plots.py \
    --results-file results/statistical_analysis.json
```

## Exploring with Jupyter

Launch Jupyter to explore the data interactively:

```bash
jupyter notebook
```

Then create a new notebook and try:

```python
import sys
sys.path.append('.')

from src.data_collection import CoupleDataCollector
from src.facial_analysis import FaceDetector, FaceEmbeddingExtractor
from src.statistical_analysis import SimilarityAnalyzer
from src.visualization import SimilarityVisualizer

# Load your data
collector = CoupleDataCollector()
print(f"Dataset contains {collector.get_couple_count()} couples")

# Test face detection
detector = FaceDetector(method="opencv")
faces = detector.detect_faces("path/to/test/image.jpg")
print(f"Detected {len(faces)} faces")
```

## Troubleshooting

### Issue: "No module named 'cv2'"
**Solution**: Install opencv-python
```bash
pip install opencv-python
```

### Issue: "No faces detected"
**Solutions**:
1. Try a different detector: `--detector mtcnn`
2. Check image quality and face visibility
3. Ensure proper lighting in images

### Issue: TensorFlow errors
**Solution**:
- For CPU only: `pip install tensorflow-cpu`
- For GPU: Install CUDA and cuDNN first
- For M1/M2 Mac: Use `tensorflow-macos` and `tensorflow-metal`

### Issue: Out of memory
**Solution**:
- Process couples in smaller batches
- Use a machine with more RAM
- Reduce image sizes

### Issue: Slow processing
**Solutions**:
- Use `--detector opencv` (fastest)
- Process fewer couples initially
- Use GPU acceleration if available

## Dataset Recommendations

For best results, aim for:
- **Minimum**: 50 couples (for preliminary analysis)
- **Good**: 100-200 couples (for reliable statistics)
- **Excellent**: 500+ couples (for publication-quality results)

Quality criteria:
- Clear, front-facing photos
- Good lighting
- Minimal occlusions (sunglasses, hats, etc.)
- Similar image quality across dataset
- Diverse demographics

## Output Files

After running the pipeline, you'll have:

```
results/
├── statistical_analysis.json    # All statistical results
└── figures/
    ├── distribution_comparison.png
    ├── violin_comparison.png
    ├── box_comparison.png
    └── statistical_summary.png
```

## Next Steps

1. Review results in `results/statistical_analysis.json`
2. Check visualizations in `results/figures/`
3. Update `blog_post.md` with your findings
4. Share on social media, blog, or portfolio
5. Add project to resume/GitHub

## Citation

If you use this code or methodology, please cite:

```
[Your Name]. (2024). Do We Date Alike? A Data-Driven Investigation
of Facial Similarity in Romantic Couples. GitHub repository:
https://github.com/yourusername/do-we-date-alike
```

## Support

For issues or questions:
- Open an issue on GitHub
- Check the README.md for more information
- Review the code documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
