# MIRA Detection Pipeline

## Summary

MIRA (Microplastic Intelligent Recognition Architecture) detects and segments microplastic particles in darkfield microscopy images using a two-stage pipeline. In the first stage, a YOLO object-detection model scans the full image and produces bounding boxes around candidate particles along with a confidence score for each. In the second stage, the SAM 2 (Segment Anything Model 2) foundation model receives each bounding box as a spatial prompt and generates a pixel-precise segmentation mask that delineates the exact boundary of the particle. When a CNN-predicted chemical map is available for the image, the pipeline additionally classifies each segmented particle by material type (e.g. Polystyrene, Polymethyl Methacrylate, Polyethylene) by mapping the mask onto the chemical grid and taking a majority vote of the non-background cells it overlaps. The result is a set of per-particle records, each containing a bounding box, a binary segmentation mask, a detection confidence, and an optional material label.

---

## Detailed Pipeline Description

### 1. Image Acquisition

Input images are darkfield optical microscopy captures, typically at 100x magnification under 532 nm laser illumination. Samples are prepared on membrane filters (e.g. 0.4 um PCTE or 0.2 um AAO) and gold-coated for enhanced contrast. Images are stored as JPEG or PNG files and may be organized into YOLO-format dataset directories with `images/` and `labels/` subfolders for training, validation, and testing splits.

### 2. Stage 1 -- YOLO Object Detection

The first stage uses a custom-trained YOLO model (the "hunter-yolo" family) to localize particles. MIRA supports multiple model versions (e.g. v0.2.3 for bounding-box detection, v0.3.2 for segmentation, v0.4.4 for the latest bounding-box iteration) and can auto-discover versions from a models directory by filename convention (`hunter-yolo-v<version>.pt`).

**Inference parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf` | 0.3 | Minimum confidence threshold; predictions below this score are discarded. |
| `iou` | 0.25 | IoU threshold for agnostic non-maximum suppression (NMS), which removes duplicate overlapping boxes regardless of class. |
| `imgsz` | 1280 | Input image is resized to this dimension (longest side) before inference, preserving aspect ratio. |

YOLO processes the image in a single forward pass and outputs, for each detected particle:

- **Bounding box** in `[x1, y1, x2, y2]` pixel coordinates (top-left and bottom-right corners).
- **Confidence score** between 0 and 1 indicating the model's certainty that the region contains a particle.
- **Class ID and name** (though in the current single-class setup, all detections are "microplastic").

### 3. Stage 2 -- SAM 2 Instance Segmentation

Each bounding box from Stage 1 is fed as a box prompt into SAM 2 (Segment Anything Model 2), a vision-foundation model pre-trained on large-scale segmentation data. SAM 2 produces a high-resolution binary mask for the object inside the box.

**Box shrinking.** Before prompting SAM 2, each bounding box is shrunk inward by a configurable fraction (default 10%) of its width and height. This removes the margin of background pixels that YOLO tends to include at box edges, which would otherwise confuse SAM 2 into segmenting background regions. The shrink is clamped so that the box never collapses to zero area. Mathematically, for a box `[x1, y1, x2, y2]` with width `w` and height `h`:

```
x1' = x1 + w * frac
y1' = y1 + h * frac
x2' = x2 - w * frac
y2' = y2 - h * frac
```

All coordinates are clipped to image bounds.

**Mask generation.** SAM 2 is run in single-mask mode (`multimask_output=False`) for each box prompt independently. The image encoder processes the full image once via `set_image()`, and then the lightweight mask decoder is invoked per box. The output is a 2-D logit array at the original image resolution. A threshold of 0.5 is applied to produce a binary mask.

### 4. Chemical Material Classification (Optional)

When a CNN-predicted chemical map is paired with the image, the pipeline determines the material composition of each segmented particle. The chemical map is a CSV file structured as a 2-D grid where each cell contains either a material name (e.g. "Polystyrene", "Polymethyl Methacrylate", "Polyethylene", "Cotton") or "NA" for background.

**Spatial mapping.** The grid maps uniformly onto the image: for an image of dimensions `H x W` pixels and a grid of `R` rows by `C` columns, each grid cell covers a rectangular region of `(H/R) x (W/C)` pixels. A mask pixel at coordinates `(x, y)` maps to grid cell `(row, col)` by:

```
row = floor(y / (H / R))    clamped to [0, R-1]
col = floor(x / (W / C))    clamped to [0, C-1]
```

**Majority vote.** The pipeline collects all unique grid cells that the mask overlaps, discards cells labeled "NA", and counts how many cells vote for each material. The material with the most votes is assigned to that particle. If all overlapping cells are "NA", the particle is labeled "NA" (unknown material).

### 5. Evaluation Metrics

When ground-truth segmentation labels are available (in YOLO polygon format), the pipeline evaluates predictions using greedy bipartite mask matching:

1. **IoU matrix.** An IoU (Intersection over Union) score is computed between every predicted mask and every ground-truth mask.

2. **Greedy matching.** The pair with the highest IoU is matched first, then removed from consideration. This repeats until no unmatched masks remain on either side.

3. **Classification.** Each matched pair is classified as a true positive (TP) if its IoU exceeds a threshold (default 0.5), or as a false negative (FN) otherwise. Unmatched predictions are false positives (FP). Unmatched ground-truth masks are additional false negatives.

4. **Aggregate metrics.** Per-image metrics (mean IoU, TP, FP, FN) are rolled up into dataset-level precision, recall, and F1:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

### 6. Pipeline Modes

MIRA can be used in three modes depending on which stages are needed:

| Mode | Class | What it does |
|------|-------|-------------|
| YOLO only | `MiraYOLO` | Runs Stage 1 only. Returns bounding boxes and confidences. Useful for rapid particle counting without segmentation. |
| SAM 2 only | `MiraSAM2` | Runs Stage 2 only. Requires externally provided bounding boxes as input. Useful when boxes come from a different detector or manual annotation. |
| Sequential | `MiraSequential` | Runs Stage 1 then Stage 2 end-to-end. Returns boxes, masks, and confidences in a single call. Supports batch inference and dataset-level evaluation. |

### 7. Configuration Reference

**YOLO model versions:**

| Version | Type | Notes |
|---------|------|-------|
| v0.2.x | Bounding box | Early detection models |
| v0.3.x | Segmentation | Trained with YOLO-seg for direct mask output |
| v0.4.x | Bounding box | Latest detection models, paired with SAM 2 for segmentation |

**SAM 2 checkpoints:**

| Checkpoint | Architecture |
|------------|-------------|
| `sam2.1_hiera_large.pt` | Hiera-Large backbone, SAM 2.1 |
| `sam2_hiera_l.pt` | Hiera-Large backbone, SAM 2.0 |

**Key thresholds:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Confidence threshold | 0.1 -- 0.3 | Controls sensitivity vs. false positive rate |
| NMS IoU threshold | 0.25 | Suppresses duplicate overlapping detections |
| Box shrink fraction | 0.10 | Tightens YOLO boxes before SAM 2 prompting |
| Mask binarization threshold | 0.5 | Converts SAM 2 logits to binary masks |
| Evaluation IoU threshold | 0.5 | Minimum IoU for a prediction to count as a true positive |
