from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Suppress verbose framework logs during evaluation
warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "False"
logging.getLogger("ultralytics").setLevel(logging.ERROR)


#constants
#COCO class IDs used from the base YOLOv8n model
COCO_PERSON_CLASS = 0
COCO_BICYCLE_CLASS = 1
COCO_MOTORCYCLE_CLASS = 3

#Confidence thresholds 
TWO_WHEELER_CONF = 0.30
PERSON_CONF = 0.30
HELMET_CONF = 0.40
LICENSE_PLATE_CONF = 0.25   # Low threshold to catch partial plates

#IoU threshold for rider–vehicle vertical association
RIDER_VEHICLE_IOU_THRESHOLD = 0.05   #loose — riders sit above, not on top of vehicle
RIDER_VEHICLE_VERTICAL_MARGIN = 0.35  #fraction of vehicle height to extend upward

#Maximum edit distance ratio for OCR post-filter
#If output is all non-alphanumeric, it's rejected
MIN_ALNUM_RATIO = 0.5

# Image pre-processing for OCR
OCR_UPSCALE_FACTOR = 3          #Upscale LP crop for better OCR
OCR_MIN_SIDE_PX = 64            #If shorter side < this, upscale more aggressively

# Helmet model class mapping 
HELMET_CLASS_SAFE = 1
HELMET_CLASS_VIOLATION = 0


#helper utilities
def _box_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def _horizontal_overlap_ratio(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Fraction of boxA's width that horizontally overlaps with boxB.
    Used to check if a person is 'above' a vehicle.
    """
    overlapX = max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0]))
    widthA = max(1, boxA[2] - boxA[0])
    return overlapX / widthA


def _preprocess_license_plate(crop: np.ndarray) -> np.ndarray:
    #apply image processing to improve OCR accuracy on license plate crops.
  
    h, w = crop.shape[:2]
    scale = OCR_UPSCALE_FACTOR
    # Ensure minimum size
    if min(h, w) < OCR_MIN_SIDE_PX:
        scale = max(scale, int(np.ceil(OCR_MIN_SIDE_PX / min(h, w))))

    resized = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

    # Adaptive threshold works better than global for license plates in varied lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def _clean_ocr_text(raw: str) -> str:
    """
    Post-process OCR output:
    """
    if not raw:
        return "UNKNOWN"
    #keep alphanumeric and hyphens only
    cleaned = re.sub(r"[^A-Za-z0-9\-]", "", raw).upper()
    if not cleaned:
        return "UNKNOWN"
    #reject if fewer than MIN_ALNUM_RATIO fraction are alphanumeric
    alnum_count = sum(c.isalnum() for c in cleaned)
    if alnum_count / max(len(cleaned), 1) < MIN_ALNUM_RATIO:
        return "UNKNOWN"
    return cleaned


def _safe_crop(image: np.ndarray, box: np.ndarray, padding: int = 4) -> np.ndarray:
    """
    Crop image to bounding box with optional padding. Clamps to image bounds.
    """
    H, W = image.shape[:2]
    x1 = max(0, int(box[0]) - padding)
    y1 = max(0, int(box[1]) - padding)
    x2 = min(W, int(box[2]) + padding)
    y2 = min(H, int(box[3]) + padding)
    if x2 <= x1 or y2 <= y1:
        return image  # degenerate crop fallback
    return image[y1:y2, x1:x2]


def _extend_box_upward(box: np.ndarray, margin_fraction: float, image_height: int) -> np.ndarray:
    """
    Extend a vehicle's bounding box upward by `margin_fraction` of its height.
    This creates a 'rider zone' above the vehicle to associate nearby pedestrians.
    """
    x1, y1, x2, y2 = box
    vehicle_height = y2 - y1
    extended_y1 = max(0, y1 - vehicle_height * margin_fraction)
    return np.array([x1, extended_y1, x2, y2], dtype=float)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TrafficViolationDetector:

    def __init__(self, model_dir: str = "./models") -> None:
        """
        Initialize and load all models here.
        model_dir: path to directory containing model weights.
        """
        # Late imports to keep top-level import overhead minimal
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from e

        try:
            import easyocr
        except ImportError as e:
            raise RuntimeError(
                "easyocr not installed. Run: pip install easyocr"
            ) from e

        model_dir = Path(model_dir)

        # Validate model files
        required = {
            "yolov8n.pt": "Two-wheeler and person detector (COCO YOLOv8n)",
            "helmet_yolov8n.pt": "Helmet violation detector (fine-tuned YOLOv8n)",
            "lp_yolov8n.pt": "License plate detector (fine-tuned YOLOv8n)",
        }
        for fname, desc in required.items():
            path = model_dir / fname
            if not path.exists():
                raise RuntimeError(
                    f"Model file not found: {path}\n"
                    f"  Required for: {desc}\n"
                    f"  See README.md → 'Model Download Instructions'"
                )

        
        #loading yolo models
        self._detector = YOLO(str(model_dir / "yolov8n.pt"))
        self._helmet_detector = YOLO(str(model_dir / "helmet_yolov8n.pt"))
        self._lp_detector = YOLO(str(model_dir / "lp_yolov8n.pt"))

        #warm-up pass to pre-compile inference graph (avoids cold-start on first predict)
        _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._detector(_dummy, verbose=False)
        self._helmet_detector(_dummy, verbose=False)
        self._lp_detector(_dummy, verbose=False)


        #load ocr models
        self._ocr = easyocr.Reader(
            ["en"],
            gpu=False,            # CPU inference for portability
            verbose=False,
            model_storage_directory=str(model_dir / "easyocr_models"),
            download_enabled=True,   # Allow on first run; disable if strictly offline
        )


    def predict(self, image_path: str) -> dict:
        """
        Input:
        image_path: Path to input image
        Output:
        {
            "violations": [
                {
                "num_riders": int,
                "helmet_violations": int,
                "license_plate": "string"
                }
            ]
        }
        """
        empty_result: dict = {"violations": []}

        #load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(
                    f"[WARNING] Cannot read image: {image_path}",
                    file=sys.stderr,
                )
                return empty_result
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Error loading image {image_path}: {exc}", file=sys.stderr)
            return empty_result

        #detect two wheelers and person
        two_wheelers = self._detect_two_wheelers(image)
        if not two_wheelers:
            return empty_result  # No bikes → no violations

        persons = self._detect_persons(image)

        #rider-vehicle association
        vehicle_riders: List[List[np.ndarray]] = self._associate_riders(
            two_wheelers, persons, image.shape[0]
        )

        #per vehicle processing
        violations = []
        for vehicle_box, riders in zip(two_wheelers, vehicle_riders):
            violation_entry = self._process_vehicle(image, vehicle_box, riders)
            if violation_entry is not None:
                violations.append(violation_entry)

        if not violations:
            return {"violations": []}

        return {"violations": violations}

    
    #helper functions
    def _detect_two_wheelers(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run COCO detector and return bounding boxes for motorcycles.

        Returns list of np.ndarray([x1, y1, x2, y2]).
        """
        try:
            results = self._detector(image, verbose=False, conf=TWO_WHEELER_CONF)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Two-wheeler detection failed: {exc}", file=sys.stderr)
            return []

        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id == COCO_MOTORCYCLE_CLASS and conf >= TWO_WHEELER_CONF:
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes.append(xyxy)
        return boxes

    def _detect_persons(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run COCO detector and return bounding boxes for all persons.

        Returns list of np.ndarray([x1, y1, x2, y2]).
        """
        try:
            results = self._detector(image, verbose=False, conf=PERSON_CONF)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Person detection failed: {exc}", file=sys.stderr)
            return []

        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id == COCO_PERSON_CLASS and conf >= PERSON_CONF:
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes.append(xyxy)
        return boxes

    def _associate_riders(
        self,
        vehicles: List[np.ndarray],
        persons: List[np.ndarray],
        image_height: int,
    ) -> List[List[np.ndarray]]:
        """
        Assign each detected person to the most likely vehicle they are riding.

        Returns:
            List of length len(vehicles), where entry i is the list of person boxes
            assigned to vehicle i.
        """
        if not persons:
            return [[] for _ in vehicles]

        # Build extended zones
        extended_zones = [
            _extend_box_upward(v, RIDER_VEHICLE_VERTICAL_MARGIN, image_height)
            for v in vehicles
        ]

        # Greedy assignment: for each person find best vehicle
        assignment: List[Optional[int]] = [None] * len(persons)
        best_score: List[float] = [0.0] * len(persons)

        for p_idx, person_box in enumerate(persons):
            for v_idx, zone in enumerate(extended_zones):
                iou = _box_iou(person_box, zone)
                h_overlap = _horizontal_overlap_ratio(person_box, vehicles[v_idx])
                # Combined score: IoU of extended zone + horizontal overlap bonus
                score = iou + 0.2 * h_overlap
                if iou >= RIDER_VEHICLE_IOU_THRESHOLD and score > best_score[p_idx]:
                    best_score[p_idx] = score
                    assignment[p_idx] = v_idx

        # Group persons by vehicle
        vehicle_riders: List[List[np.ndarray]] = [[] for _ in vehicles]
        for p_idx, v_idx in enumerate(assignment):
            if v_idx is not None:
                vehicle_riders[v_idx].append(persons[p_idx])

        return vehicle_riders


    def _process_vehicle(
        self,
        image: np.ndarray,
        vehicle_box: np.ndarray,
        riders: List[np.ndarray],
    ) -> Optional[dict]:
        """
        Determine violations for a single vehicle.

        Returns a violation dict if any violation is found, else None.
        """
        num_riders = len(riders)
        helmet_violations = 0

        if num_riders > 0:
            helmet_violations = self._count_helmet_violations(image, riders)

        # --- Violation check ---
        overcrowded = num_riders > 2
        helmet_issue = helmet_violations > 0

        if not (overcrowded or helmet_issue):
            return None  # Clean vehicle — skip

        # --- License plate ---
        license_plate = self._get_license_plate(image, vehicle_box)

        return {
            "num_riders": num_riders,
            "helmet_violations": helmet_violations,
            "license_plate": license_plate,
        }

    def _count_helmet_violations(
        self, image: np.ndarray, riders: List[np.ndarray]
    ) -> int:
        
        violations = 0
        for rider_box in riders:
            head_crop = self._get_head_crop(image, rider_box)
            if head_crop is None or head_crop.size == 0:
                continue
            try:
                results = self._helmet_detector(
                    head_crop, verbose=False, conf=HELMET_CONF
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARNING] Helmet detection failed: {exc}", file=sys.stderr)
                continue

            has_helmet = False
            no_helmet_detected = False

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if conf < HELMET_CONF:
                        continue
                    if cls_id == HELMET_CLASS_SAFE:
                        has_helmet = True
                    elif cls_id == HELMET_CLASS_VIOLATION:
                        no_helmet_detected = True

            if no_helmet_detected:
                violations += 1

        return violations

    def _get_head_crop(
        self, image: np.ndarray, rider_box: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Crop the upper 40% of a rider's bounding box as the 'head region'.
        """
        x1, y1, x2, y2 = rider_box
        rider_height = y2 - y1
        head_y2 = y1 + rider_height * 0.45  # top 45% of rider
        head_box = np.array([x1, y1, x2, head_y2])
        return _safe_crop(image, head_box, padding=6)

    def _get_license_plate(
        self, image: np.ndarray, vehicle_box: np.ndarray
    ) -> str:
        """
        Detect and read the license plate on a vehicle.
        Returns 'UNKNOWN' on any failure.
        """
        vehicle_crop = _safe_crop(image, vehicle_box, padding=8)
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "UNKNOWN"

        #lp detection
        try:
            lp_results = self._lp_detector(
                vehicle_crop, verbose=False, conf=LICENSE_PLATE_CONF
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] LP detection failed: {exc}", file=sys.stderr)
            return "UNKNOWN"

        best_lp_box: Optional[np.ndarray] = None
        best_conf = 0.0
        for result in lp_results:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                if conf >= LICENSE_PLATE_CONF and conf > best_conf:
                    best_lp_box = box.xyxy[0].cpu().numpy()
                    best_conf = conf

        if best_lp_box is None:
            # Fallback: try OCR on the bottom 25% of the vehicle crop directly
            # (plates are usually at the bottom)
            return self._ocr_bottom_strip(vehicle_crop)

        lp_crop = _safe_crop(vehicle_crop, best_lp_box, padding=4)
        if lp_crop is None or lp_crop.size == 0:
            return "UNKNOWN"

        return self._run_ocr(lp_crop)

    def _ocr_bottom_strip(self, vehicle_crop: np.ndarray) -> str:
        """
        Fallback: if no LP bounding box detected, attempt OCR on the
        bottom 25% strip of the vehicle crop where plates typically appear.
        """
        h = vehicle_crop.shape[0]
        strip = vehicle_crop[int(h * 0.72):, :]
        if strip.size == 0:
            return "UNKNOWN"
        return self._run_ocr(strip)

    def _run_ocr(self, crop: np.ndarray) -> str:
        """
        Preprocess the license plate crop and run EasyOCR.
        Concatenates all detected text fragments and cleans the result.
        """
        if crop is None or crop.size == 0:
            return "UNKNOWN"

        processed = _preprocess_license_plate(crop)

        try:
            ocr_results = self._ocr.readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                batch_size=1,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] OCR failed: {exc}", file=sys.stderr)
            return "UNKNOWN"

        if not ocr_results:
            # Second attempt: try on raw crop (sometimes adaptive thresh is too aggressive)
            try:
                ocr_results = self._ocr.readtext(
                    crop,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                    batch_size=1,
                )
            except Exception:  # noqa: BLE001
                return "UNKNOWN"

        if not ocr_results:
            return "UNKNOWN"

        #sort detections left-to-right by x-coordinate of bounding box centroid
        ocr_results_sorted = sorted(
            ocr_results,
            key=lambda r: (r[0][0][0] + r[0][2][0]) / 2  # average x
        )

        #concatenate text fragments above confidence threshold
        fragments = []
        for (_, text, conf) in ocr_results_sorted:
            if conf >= 0.3:
                fragments.append(text.strip())

        combined = "".join(fragments)
        return _clean_ocr_text(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detector — CLI utility"
    )
    parser.add_argument(
        "--model-dir",
        default="./models",
        help="Directory containing model weights (default: ./models)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image for inference",
    )
    args = parser.parse_args()

    if args.image:
        import json
        detector = TrafficViolationDetector(model_dir=args.model_dir)
        result = detector.predict(args.image)
        print(json.dumps(result, indent=2))
