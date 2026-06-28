# Traffic Rule Violation Detection System

## Overview

This system detects traffic rule violations involving two-wheelers from street images. It identifies:
- **Overcrowding** — more than 2 riders on a single vehicle.
- **Helmet violations** — riders not wearing helmets.
- **Combined violations** — both conditions simultaneously.

For each violating vehicle, it extracts the **license plate text** using OCR.

---

## Pipeline Architecture

```
Input Image
     │
     ▼
┌─────────────────────────┐
│  Stage 1: Two-Wheeler   │  YOLOv8n (COCO pretrained)
│  Detection              │  Classes: motorcycle (3), bicycle filtered
└────────────┬────────────┘
             │  Cropped vehicle ROIs
             ▼
┌─────────────────────────┐
│  Stage 2: Person        │  YOLOv8n (COCO pretrained)
│  Detection (full image) │  Class: person (0)
└────────────┬────────────┘
             │  Person bboxes
             ▼
┌─────────────────────────┐
│  Stage 3: Rider–Vehicle │  IoU / vertical proximity
│  Association            │  Hungarian matching via scipy
└────────────┬────────────┘
             │  Per-vehicle rider lists
             ▼
┌─────────────────────────┐
│  Stage 4: Helmet        │  YOLOv8n fine-tuned on helmet dataset
│  Detection              │  (helmet / no-helmet classes)
└────────────┬────────────┘
             │  Helmet status per rider
             ▼
┌─────────────────────────┐
│  Stage 5: Violation     │  Rule: riders > 2 OR helmet_viol > 0
│  Filtering              │
└────────────┬────────────┘
             │  Violating vehicles only
             ▼
┌─────────────────────────┐
│  Stage 6: License Plate │  YOLOv8n fine-tuned on LP dataset
│  Detection              │
└────────────┬────────────┘
             │  LP ROI crops
             ▼
┌─────────────────────────┐
│  Stage 7: OCR           │  EasyOCR (en) with preprocessing
│  (EasyOCR)              │  Upscale → Denoise → Threshold → Read
└────────────┬────────────┘
             │
             ▼
        JSON Output
```

---

## Model Choices & Rationale

| Model | Purpose | Size | Why |
|---|---|---|---|
| `yolov8n.pt` | Two-wheeler + person detection | ~6 MB | Smallest YOLOv8; fast; COCO pretrained covers motorcycles & persons |
| `helmet_yolov8n.pt` | Helmet / no-helmet classification | ~6 MB | Fine-tuned YOLOv8n; sourced from [meryemsakin/helmet-detection-yolov8](https://github.com/meryemsakin/helmet-detection-yolov8) |
| `lp_yolov8n.pt` | License plate localization | ~6 MB | Fine-tuned YOLOv8n; sourced from [benjnb/yolo-license-plate-detection](https://www.kaggle.com/code/benjnb/yolo-license-plate-detection) (Kaggle) |
| EasyOCR `en` | License plate text recognition | ~40 MB | Ships offline; supports alphanumeric; robust to blur |

**Total: ~58 MB** — well within the 250 MB constraint.

All models are loaded once in `__init__` and reused across `predict()` calls for efficiency.

---

## Directory Structure

```
ROLL_NUMBER/
├── solution.py              # Main implementation
├── requirements.txt         # All Python dependencies
├── README.md                # This file
└── models/
    └── easyocr_models/
        ├── craft_mlt_25k.pth
        ├── english_g2.pth
    ├── yolov8n.pt           # COCO detection model (vehicles + persons)
    ├── helmet_yolov8n.pt    # Helmet detector (fine-tuned)
    └── lp_yolov8n.pt        # License plate detector (fine-tuned)
```

---

## Setup Instructions

### 1. Create & activate a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate.bat      # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `torch` pulls in ~700 MB by default. For CPU-only install (smaller):
> ```bash
> pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## Running Inference

### Single image
```python
from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="./models")
result = detector.predict("path/to/image.jpg")
print(result)
```

### Batch inference (example script)
```python
import os, json
from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="./models")

image_dir = "test_images/"
for fname in os.listdir(image_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(image_dir, fname)
        result = detector.predict(path)
        print(f"{fname}: {json.dumps(result, indent=2)}")
```

### Download models then run
```bash
python solution.py --download-models        # one-time setup
python solution.py --image path/to/img.jpg  # single inference
```

---

## Output Format

```json
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 2,
      "license_plate": "MH12AB1234"
    }
  ]
}
```

- `violations` is an empty list `[]` when no violations are detected.
- `license_plate` is `"UNKNOWN"` when OCR fails or the plate is unreadable.
- Multiple violating vehicles produce multiple entries in the list.

---

## Assumptions

1. **Camera angle**: Assumes  street level or slightly elevated CCTV angle.
2. **Two-wheeler definition**: Motorcycles and scooters (COCO class `motorcycle`). Bicycles are excluded as they typically don't require helmets under traffic law.
3. **Rider–vehicle association**: Riders whose bounding boxes vertically overlap with (or sit directly above) the vehicle bounding box are assigned to that vehicle.
4. **Helmet detection granularity**: Each rider crop is independently classified. The model detects whether a helmet is worn (class 0) or not (class 1).
5. **License plate**: Only one plate is expected per vehicle. The highest-confidence plate detection is used.
6. **Violation threshold**: Any helmet absence OR rider count > 2 is flagged. Clean vehicles (all helmets, ≤ 2 riders) are excluded from output.
7. **Minimum confidence**: Detections below 0.3 confidence are discarded to reduce false positives.

---

## Limitations

1. **Occlusion**: Riders hidden behind the vehicle or each other may be undercounted.
2. **Nighttime / Low lighting**: Detection accuracy drops significantly below ~50 lux; no IR model is included.
3. **Small objects**: License plates smaller than ~20×6 pixels post-resize may not be OCR-readable.
4. **Non-standard plates**: Heavily stylized, damaged, or foreign plates may produce garbled OCR output.
5. **Crowd scenes**: Pedestrians very close to vehicles may be falsely associated as riders.
6. **Single-frame analysis**: No temporal tracking; transient occlusions are not recovered.
7. **Helmet model domain gap**: The helmet detector (sourced from [meryemsakin/helmet-detection-yolov8](https://github.com/meryemsakin/helmet-detection-yolov8)) was trained primarily on motorcycle helmet data, but edge cases like heavily tinted visors or non-standard headgear may have lower recall.

---

## Failure Case Analysis

| Scenario | Expected Behavior | Mitigation |
|---|---|---|
| No two-wheeler in image | `{"violations": []}` | Graceful empty return |
| Two-wheeler detected, no riders visible | Vehicle skipped (0 riders) | Minimum rider threshold |
| License plate not detected | `"license_plate": "UNKNOWN"` | Fallback string |
| OCR returns garbage (edit dist. too high) | `"license_plate": "UNKNOWN"` | Post-OCR regex filter |
| Image file not found / corrupt | `{"violations": []}` + stderr warning | try/except in predict() |
| Model weights missing | RuntimeError with clear message | Checked in `__init__` |

---
