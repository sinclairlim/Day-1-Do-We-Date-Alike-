# NEW FEATURE: Automatic Face Extraction from Couple Photos

## What Changed?

You can now add photos with **BOTH people in the same image**, and the system will automatically detect and extract the 2 faces!

## Before vs After

### BEFORE (Manual cropping required)
```
You had to:
1. Find a photo of a couple
2. Manually crop Person A → save as person1.jpg
3. Manually crop Person B → save as person2.jpg
4. Organize into data/raw/couple_XXXX/ folder
5. Repeat for each couple...
```

### AFTER (Automatic extraction!)
```
Now you can:
1. Have a photo with both people
2. Run: python add_couple_photos.py couple_photo.jpg
3. Done! Both faces automatically extracted and organized
```

---

## New Files Added

1. **[add_couple_photos.py](add_couple_photos.py)**
   - Command-line tool to easily add couple photos
   - Handles single photos or entire folders
   - Auto-detects and extracts 2 faces

2. **[HOW_TO_ADD_PHOTOS.md](HOW_TO_ADD_PHOTOS.md)**
   - Complete guide on using the new feature
   - Examples, troubleshooting, tips

## Modified Files

1. **[src/data_collection/collect_couples.py](src/data_collection/collect_couples.py)**
   - Added `add_couple_from_single_photo()` method
   - Automatically detects 2 faces in one photo
   - Extracts and saves both faces

2. **[README.md](README.md)**
   - Updated Quick Start section
   - Mentions new feature

3. **[QUICKSTART.md](QUICKSTART.md)**
   - Updated with new easy workflow
   - Added examples

---

## How It Works

### Architecture

```
Input Photo (couple_photo.jpg)
         ↓
   [Face Detection Algorithm]
         ↓
    Finds 2 faces
    (x1,y1,w1,h1) and (x2,y2,w2,h2)
         ↓
   [Face Extraction]
         ↓
  Crops each face with padding
         ↓
   [Normalization]
         ↓
  Resizes to 160x160
         ↓
   [Save]
         ↓
  data/raw/couple_XXXX/person1.jpg
  data/raw/couple_XXXX/person2.jpg
```

### Technical Details

**Phase 2 (Face Detection):**
- Detects ALL faces in the image
- If exactly 2 found → perfect, use both
- If more than 2 → selects the 2 largest faces
- If 0 or 1 → error, cannot process

**Face Extraction:**
- Adds 20% padding around each face
- Ensures boundaries don't exceed image limits
- Resizes to standard 160x160 pixels
- Saves as JPEG

---

## Usage Examples

### 1. Single Photo
```bash
python add_couple_photos.py celebrity_couple.jpg
```

### 2. Batch Processing
```bash
python add_couple_photos.py my_couples_folder/
```

### 3. Different Detector
```bash
# Use MTCNN for better accuracy
python add_couple_photos.py photo.jpg --detector mtcnn
```

### 4. Programmatic (Python)
```python
from src.data_collection import CoupleDataCollector

collector = CoupleDataCollector()
collector.add_couple_from_single_photo(
    "couple_photo.jpg",
    detector_method="opencv"
)
```

---

## Error Handling

The system handles various edge cases:

### No Faces Detected
```
✗ Error: No faces detected in photo.jpg
```
**Solution**: Check photo quality, lighting, or try `--detector mtcnn`

### Only 1 Face
```
✗ Error: Only 1 face detected. Need exactly 2 faces.
```
**Solution**: Ensure both people are visible in the photo

### 3+ Faces
```
⚠ Warning: Found 4 faces. Using the 2 largest.
✓ Successfully added couple_0001
```
**Solution**: Automatically handled! Check results to verify correct people.

---

## Benefits

### For Users
- ✓ **Much Faster**: No manual cropping needed
- ✓ **Less Error-Prone**: Automated process
- ✓ **Batch Processing**: Handle 50+ photos at once
- ✓ **Consistent Results**: Same extraction for all

### For Data Collection
- ✓ Easier to gather data from web (most couple photos have both people)
- ✓ Can use celebrity couple photos directly
- ✓ Can process entire datasets quickly

### For Analysis
- ✓ Standardized face extraction (160x160)
- ✓ Consistent padding and alignment
- ✓ Better comparison across couples

---

## Photo Requirements

### ✓ Works Best With
- Clear, front-facing or near-front photos
- Good lighting
- Both faces clearly visible
- Faces are primary subjects (not tiny)
- Minimal occlusions

### ⚠️ May Have Issues With
- Very dark/low-light photos
- Extreme profile views
- Faces heavily obscured (sunglasses, hats)
- Multiple people (will use 2 largest)

---

## Detector Options

### opencv (default)
- **Speed**: ⚡⚡⚡ Very Fast
- **Accuracy**: ⭐⭐ Good
- **Best for**: Clear, front-facing photos
- **Use when**: Processing many photos quickly

### mtcnn
- **Speed**: ⚡⚡ Moderate
- **Accuracy**: ⭐⭐⭐ Excellent
- **Best for**: Varied angles, difficult lighting
- **Use when**: Default detector misses faces

### dlib
- **Speed**: ⚡ Slower
- **Accuracy**: ⭐⭐⭐ Very Good
- **Best for**: Traditional, reliable detection
- **Use when**: Need alternative to OpenCV/MTCNN

---

## Integration with Pipeline

The new feature integrates seamlessly:

```bash
# Old workflow:
1. Manually crop photos
2. Organize files
3. python run_pipeline.py

# New workflow:
1. python add_couple_photos.py photos/
2. python run_pipeline.py
```

Everything else remains the same:
- Embedding extraction works as before
- Statistical analysis unchanged
- Visualizations unchanged
- Results format identical

---

## Backward Compatibility

✓ **Fully backward compatible!**

Old methods still work:
```python
# Still works - separate files
collector.add_couple_from_files(
    "person1.jpg",
    "person2.jpg"
)

# Still works - manual organization
data/raw/couple_0001/person1.jpg
data/raw/couple_0001/person2.jpg
```

New method is just an additional option!

---

## Testing Checklist

If you want to test this feature:

- [ ] Try single photo: `python add_couple_photos.py test.jpg`
- [ ] Try batch: `python add_couple_photos.py test_folder/`
- [ ] Check extracted faces in `data/raw/couple_XXXX/`
- [ ] Verify both faces look correct
- [ ] Try different detectors if needed
- [ ] Run pipeline: `python run_pipeline.py`
- [ ] Verify analysis works with extracted faces

---

## Future Enhancements

Potential improvements:
- [ ] Add face quality scoring
- [ ] Allow user to choose which 2 faces (if 3+ detected)
- [ ] Add preview/confirmation before saving
- [ ] Support for video frames
- [ ] Web interface for upload

---

## Documentation Updated

All documentation has been updated:
- ✓ [README.md](README.md) - Quick start section
- ✓ [QUICKSTART.md](QUICKSTART.md) - New workflow
- ✓ [HOW_TO_ADD_PHOTOS.md](HOW_TO_ADD_PHOTOS.md) - Complete guide
- ✓ [add_couple_photos.py](add_couple_photos.py) - Script with help text

---

## Questions?

See:
- [HOW_TO_ADD_PHOTOS.md](HOW_TO_ADD_PHOTOS.md) for detailed guide
- [QUICKSTART.md](QUICKSTART.md) for quick reference
- Run `python add_couple_photos.py --help` for command options

---

**This feature makes data collection 10x easier!** 🎉

Instead of manually cropping 100 photos (1-2 hours), just run:
```bash
python add_couple_photos.py my_couples_folder/
```
Done in minutes!
