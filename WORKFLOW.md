# Project Workflow

## Visual Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DO WE DATE ALIKE?                                 │
│                    Research Pipeline Flow                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA COLLECTION                                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: Couple Images                                                   │
│    │                                                                     │
│    ├─→ Celebrity couples (public sources)                              │
│    ├─→ Research datasets (LFW, CelebA)                                 │
│    └─→ Personal data (with permission)                                 │
│                                                                          │
│  Module: src/data_collection/collect_couples.py                        │
│    │                                                                     │
│    └─→ CoupleDataCollector                                             │
│         • add_couple_from_files()                                       │
│         • add_couple_from_urls()                                        │
│         • Organizes in data/raw/                                        │
│                                                                          │
│  Output: data/raw/                                                      │
│    ├── couple_0001/                                                     │
│    │   ├── person1.jpg                                                  │
│    │   └── person2.jpg                                                  │
│    └── metadata.json                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: FACE DETECTION                                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Module: src/facial_analysis/face_detector.py                          │
│    │                                                                     │
│    └─→ FaceDetector                                                     │
│         • detect_faces()      ← Locate faces in image                  │
│         • extract_face()      ← Crop & align face                      │
│         • extract_and_save()  ← Save processed face                    │
│                                                                          │
│  Algorithms Available:                                                  │
│    ├─→ OpenCV Haar Cascade (fast, lightweight)                        │
│    ├─→ MTCNN (accurate, robust to angles)                             │
│    └─→ dlib (traditional, reliable)                                    │
│                                                                          │
│  Processing:                                                            │
│    Original Image → Face Detection → Alignment → 160x160 normalized    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: EMBEDDING EXTRACTION                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Module: src/facial_analysis/embedding_extractor.py                    │
│    │                                                                     │
│    └─→ FaceEmbeddingExtractor                                          │
│         • extract_embedding()  ← Face → 128-D vector                   │
│         • compute_similarity() ← Compare embeddings                     │
│                                                                          │
│  Deep Learning Models:                                                  │
│    ├─→ FaceNet (128-dim, fast, recommended)                           │
│    ├─→ VGGFace (2622-dim, accurate)                                   │
│    └─→ ArcFace (512-dim, state-of-the-art)                            │
│                                                                          │
│  Process:                                                               │
│    Face Image → CNN → Embedding Vector [0.123, -0.456, ...]           │
│                                                                          │
│  Similarity Calculation:                                                │
│    cosine_similarity = (A · B) / (||A|| × ||B||)                       │
│    Range: [-1, 1], Higher = More Similar                               │
│                                                                          │
│  Output: data/processed/                                                │
│    ├── couple_0001/                                                     │
│    │   ├── person1_embedding.npy                                        │
│    │   └── person2_embedding.npy                                        │
│    └── embeddings_results.json                                         │
│         {"couple_id": "couple_0001", "similarity": 0.723, ...}         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: STATISTICAL ANALYSIS                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Module: src/statistical_analysis/analyze_similarity.py                │
│    │                                                                     │
│    └─→ SimilarityAnalyzer                                              │
│                                                                          │
│  Step 1: Load Couple Data                                              │
│    • Read embeddings_results.json                                      │
│    • Extract couple similarity scores                                  │
│                                                                          │
│  Step 2: Generate Baseline                                             │
│    • Create random pairs from same faces                               │
│    • Calculate random pair similarities                                │
│    • Ensure fair comparison                                            │
│                                                                          │
│  Step 3: Descriptive Statistics                                        │
│    Couples:        mean=X.XX, std=X.XX, n=N                           │
│    Random Pairs:   mean=Y.YY, std=Y.YY, n=N                           │
│    Difference:     X.XX - Y.YY = Z.ZZ                                  │
│                                                                          │
│  Step 4: Hypothesis Testing                                            │
│    H₀: μ_couples = μ_random (no difference)                            │
│    H₁: μ_couples > μ_random (couples more similar)                     │
│    α = 0.05 (significance level)                                        │
│                                                                          │
│    Tests Performed:                                                     │
│    ├─→ Independent t-test (parametric)                                │
│    │    t-statistic, p-value                                           │
│    │                                                                    │
│    ├─→ Mann-Whitney U test (non-parametric)                           │
│    │    U-statistic, p-value                                           │
│    │                                                                    │
│    └─→ Permutation test (10,000 iterations)                           │
│         Empirical p-value                                              │
│                                                                          │
│  Step 5: Effect Size                                                    │
│    Cohen's d = (μ₁ - μ₂) / pooled_σ                                    │
│    Interpretation:                                                      │
│      |d| < 0.2: Small effect                                           │
│      |d| = 0.2-0.5: Medium effect                                      │
│      |d| = 0.5-0.8: Large effect                                       │
│      |d| > 0.8: Very large effect                                      │
│                                                                          │
│  Output: results/statistical_analysis.json                             │
│    {                                                                    │
│      "descriptive_statistics": {...},                                  │
│      "t_test": {"p_value": 0.0123, "significant": true},              │
│      "cohens_d": 0.456,                                                │
│      ...                                                                │
│    }                                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: VISUALIZATION                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Module: src/visualization/create_plots.py                             │
│    │                                                                     │
│    └─→ SimilarityVisualizer                                            │
│                                                                          │
│  Plots Generated:                                                       │
│                                                                          │
│  1. Distribution Comparison (Histogram)                                │
│     ┌─────────────────────────┐                                        │
│     │  Overlapping histograms  │                                        │
│     │  Random (blue) vs        │                                        │
│     │  Couples (red)           │                                        │
│     │  + Mean lines            │                                        │
│     └─────────────────────────┘                                        │
│                                                                          │
│  2. Violin Plot                                                         │
│     ┌─────────────────────────┐                                        │
│     │  Density distributions   │                                        │
│     │  Side-by-side            │                                        │
│     │  Shows full shape        │                                        │
│     └─────────────────────────┘                                        │
│                                                                          │
│  3. Box Plot                                                            │
│     ┌─────────────────────────┐                                        │
│     │  Quartiles & outliers    │                                        │
│     │  Median comparison       │                                        │
│     │  Sample sizes shown      │                                        │
│     └─────────────────────────┘                                        │
│                                                                          │
│  4. Statistical Summary Dashboard                                       │
│     ┌─────────────────────────┐                                        │
│     │  4-panel figure:         │                                        │
│     │  • Mean comparison       │                                        │
│     │  • Sample sizes          │                                        │
│     │  • P-values              │                                        │
│     │  • Effect size           │                                        │
│     └─────────────────────────┘                                        │
│                                                                          │
│  Output: results/figures/                                               │
│    ├── distribution_comparison.png                                     │
│    ├── violin_comparison.png                                           │
│    ├── box_comparison.png                                              │
│    └── statistical_summary.png                                         │
│                                                                          │
│  All plots: 300 DPI, publication-ready                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: PUBLICATION                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Template: blog_post.md                                                 │
│                                                                          │
│  Structure:                                                             │
│    1. Introduction                                                      │
│       • Hook the reader                                                 │
│       • State the question                                              │
│                                                                          │
│    2. Methodology                                                       │
│       • Data collection                                                 │
│       • Face detection                                                  │
│       • Embedding extraction                                            │
│       • Statistical testing                                             │
│                                                                          │
│    3. Results                                                           │
│       • Descriptive statistics                                          │
│       • Visualizations                                                  │
│       • Statistical significance                                        │
│       • Effect size                                                     │
│                                                                          │
│    4. Interpretation                                                    │
│       • What this means                                                 │
│       • Evolutionary perspective                                        │
│       • Social psychology perspective                                   │
│                                                                          │
│    5. Limitations & Future Work                                        │
│       • Sample size                                                     │
│       • Confounds                                                       │
│       • Extensions                                                      │
│                                                                          │
│    6. Conclusion                                                        │
│       • Key takeaways                                                   │
│       • Broader implications                                            │
│                                                                          │
│  Distribution:                                                          │
│    ├─→ GitHub repository                                               │
│    ├─→ Blog post (Medium/personal)                                     │
│    ├─→ LinkedIn article                                                │
│    ├─→ Twitter/X thread                                                │
│    └─→ Portfolio website                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        AUTOMATION SCRIPT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  run_pipeline.py - One Command to Rule Them All                        │
│                                                                          │
│  $ python run_pipeline.py                                               │
│                                                                          │
│  Executes:                                                              │
│    [1/3] Extract embeddings    → data/processed/                       │
│    [2/3] Statistical analysis  → results/statistical_analysis.json     │
│    [3/3] Generate plots        → results/figures/                      │
│                                                                          │
│  Optional flags:                                                        │
│    --detector {opencv|mtcnn|dlib}                                      │
│    --model {facenet|vggface|arcface}                                   │
│    --skip-extraction                                                    │
│    --skip-analysis                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

## Decision Tree: Which Detector/Model?

```
Need speed? → Use opencv + facenet (fastest)
     │
     └─ Good lighting & clear faces? → opencv
     └─ Varied angles/poses? → mtcnn
     └─ Best accuracy? → dlib

Need accuracy? → Use mtcnn + arcface (most accurate)
     │
     └─ Time is no issue? → dlib + arcface
     └─ Balance speed/accuracy? → mtcnn + facenet
```

## File Dependencies

```
run_pipeline.py
    ├─→ src/facial_analysis/extract_embeddings.py
    │       ├─→ src/facial_analysis/face_detector.py
    │       ├─→ src/facial_analysis/embedding_extractor.py
    │       └─→ src/data_collection/collect_couples.py
    │
    ├─→ src/statistical_analysis/analyze_similarity.py
    │       └─→ Uses: scipy, numpy
    │
    └─→ src/visualization/create_plots.py
            └─→ Uses: matplotlib, seaborn
```

## Data Flow Summary

```
Raw Images → Face Detection → Embeddings → Similarities → Statistics → Plots → Blog Post
                                              ↓
                                         Random Pairs
                                              ↓
                                         Comparison
```

## Success Path

```
Step 1: Personalize files (your name, links)
   ↓
Step 2: Collect 50-200 couple images
   ↓
Step 3: Run pipeline (python run_pipeline.py)
   ↓
Step 4: Fill blog_post.md with findings
   ↓
Step 5: Push to GitHub
   ↓
Step 6: Publish blog post
   ↓
Step 7: Share on social media
   ↓
Step 8: Add to portfolio
   ↓
SUCCESS: Impress employers, get engagement, land job!
```

## Timeline

```
Day 1 Morning:   Personalize & collect data (4 hours)
Day 1 Afternoon: Run pipeline & analyze (3 hours)
Day 1 Evening:   Write blog post (2 hours)
Day 2 Morning:   Publish & share (2 hours)
Day 2 Afternoon: Monitor engagement & respond

Total: 1-2 days to complete, lifetime portfolio value
```

---

**You are here**: Ready to collect data and run!

**Next step**: See [TODO.md](TODO.md) for your checklist
