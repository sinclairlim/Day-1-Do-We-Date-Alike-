# Quick Start Guide

Get up and running with "Do We Date Alike?" in 5 minutes.

## TL;DR

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add couple photos (NEW: Auto-extracts 2 faces!)
python add_couple_photos.py path/to/couple_photo.jpg
# OR add entire folder:
python add_couple_photos.py path/to/couples_folder/

# 3. Run analysis
python run_pipeline.py

# 4. Check results in results/
```

## NEW: Easy Photo Upload (Recommended!)

**Have photos with BOTH people in them?** Perfect! Just use the new auto-extractor:

```bash
# Add a single couple photo (auto-extracts 2 faces)
python add_couple_photos.py my_couple_photo.jpg

# Add all photos from a folder
python add_couple_photos.py my_couples_folder/

# Use better detector if default doesn't work well
python add_couple_photos.py photo.jpg --detector mtcnn
```

The script will:
- ✓ Automatically detect both faces
- ✓ Extract and save them separately
- ✓ Organize into the correct structure
- ✓ Handle multiple photos at once

**Requirements:**
- Photos should contain exactly 2 visible faces
- If more than 2 faces detected, the 2 largest are used
- Works best with clear, front-facing photos

## Manual Data Organization (Alternative)

If you already have separate images for each person:

```
data/raw/
├── couple_0001/
│   ├── person1.jpg
│   └── person2.jpg
├── couple_0002/
│   ├── person1.jpg
│   └── person2.jpg
└── couple_0003/
    ├── person1.jpg
    └── person2.jpg
```

## Programmatic Data Addition (Advanced)

For Python scripts:

```python
from src.data_collection import CoupleDataCollector

collector = CoupleDataCollector()

# NEW: From single photo with both people
collector.add_couple_from_single_photo(
    "path/to/couple_photo.jpg",
    detector_method="opencv"  # or "mtcnn" or "dlib"
)

# From separate image files
collector.add_couple_from_files(
    "path/to/person1.jpg",
    "path/to/person2.jpg"
)

# From URLs
collector.add_couple_from_urls(
    "https://example.com/person1.jpg",
    "https://example.com/person2.jpg"
)
```

## What You'll Get

After running the pipeline:

1. **Embeddings**: `data/processed/embeddings_results.json`
   - Facial embeddings for all couples
   - Similarity scores

2. **Statistics**: `results/statistical_analysis.json`
   - Descriptive statistics
   - T-test results
   - Effect size (Cohen's d)
   - Permutation test results

3. **Visualizations**: `results/figures/`
   - Distribution comparison
   - Violin plots
   - Box plots
   - Statistical summary

4. **Blog Post Template**: `blog_post.md`
   - Ready to fill with your findings
   - Professional structure
   - Publication-ready format

## Minimum Requirements

- **Python**: 3.9+
- **RAM**: 4GB+ (8GB+ recommended)
- **Dataset**: 50+ couples (100+ recommended)
- **Images**: Clear, front-facing photos

## Common Commands

```bash
# Run full pipeline
python run_pipeline.py

# Run with specific detector/model
python run_pipeline.py --detector mtcnn --model facenet

# Skip already-completed steps
python run_pipeline.py --skip-extraction

# See example usage
python example_usage.py

# Run individual steps
python src/facial_analysis/extract_embeddings.py
python src/statistical_analysis/analyze_similarity.py
python src/visualization/create_plots.py
```

## Troubleshooting

### "No faces detected"
- Use better quality images
- Try different detector: `--detector mtcnn`
- Check that faces are clearly visible and front-facing

### "Module not found"
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Slow performance
- Use faster detector: `--detector opencv`
- Process fewer couples initially
- Use GPU if available

## Where to Get Data

### Public Sources (with permission):
1. **Celebrity Couples**
   - Search for "celebrity couples photos"
   - Ensure images are clearly licensed

2. **Research Datasets**
   - LFW (Labeled Faces in the Wild)
   - CelebA dataset
   - Academic datasets with proper citation

3. **Your Own Data**
   - Friends/family (with permission)
   - Public social media (with consent)
   - Always respect privacy and get consent

## Next Steps

1. **Collect more data** to improve statistical power
2. **Run the analysis** with your dataset
3. **Interpret results** using the blog post template
4. **Share your findings** on:
   - Personal blog
   - Medium/Substack
   - LinkedIn
   - GitHub
   - Twitter/X

## Help & Support

- **Full Setup**: See [SETUP.md](SETUP.md)
- **Documentation**: See [README.md](README.md)
- **Examples**: Run `python example_usage.py`
- **Issues**: Open an issue on GitHub

## Citation

```
[Your Name]. (2024). Do We Date Alike? A Data-Driven Investigation.
GitHub: https://github.com/yourusername/do-we-date-alike
```

---

**Ready?** Start collecting data and run the pipeline!

```bash
python run_pipeline.py
```
