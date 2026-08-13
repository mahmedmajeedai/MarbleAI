<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a0000,40:3d0000,70:7b1c1c,100:c0392b&height=220&section=header&text=MarbleAI&fontSize=76&fontColor=ffffff&fontAlignY=38&fontStyle=bold&desc=Pixel-Level%20Fat%20Estimation%20in%20Meat%20Slices%20via%20Computer%20Vision&descSize=16&descAlignY=62&descColor=ffcccc" width="100%"/>

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/YOLO11--seg-Instance%20Segmentation-FF4B4B?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3317?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Food%20Quality%20Control-c0392b?style=flat-square"/>
  <img src="https://img.shields.io/badge/Input-Video%20Feed-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Output-Fat%25%20Per%20Slice-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/Deployment-Production%20Line-333?style=flat-square"/>
</p>

<br/>

> **MarbleAI** is an industrial-grade computer vision system that estimates fat percentage in individual meat slices in real time — frame by frame, slice by slice — using YOLO11 instance segmentation and pixel-level color analysis. Built for production conveyor belt environments.

<br/>

**[🎬 Watch Demo](#-demo) · [🏗️ How It Works](#️-how-it-works) · [⚙️ Pipeline](#️-full-pipeline) · [🚀 Quickstart](#-quickstart) · [📁 File Reference](#-file-reference)**

</div>

---

## 🧠 The Problem this Solves

Fat content in meat slices is one of the most critical quality metrics in the food processing industry. It directly affects following:

- Nutritional labeling accuracy (legal requirement in most countries)
- Product pricing and grading (lean vs standard cuts)
- Consumer trust and regulatory compliance
- Production line throughput and consistency

**How it is done today:** A trained human inspector visually estimates fat content as slices move along a conveyor. This is subjective, slow, fatiguing, and impossible to scale.

**What MarbleAI does:** A camera positioned above the conveyor feeds frames into a YOLO11 segmentation model. Each meat slice is isolated at the pixel level. A calibrated BGR color threshold separates fat tissue (white marbling) from lean meat (red). The fat pixel count divided by the total slice pixel count gives an objective, per-slice fat percentage — stamped directly onto the output video in real time.

```
Human inspector:  Visual estimate → "looks about 20%" → inconsistent, unscalable
MarbleAI:         Segment → Threshold → Count → 18.74% → objective, per frame
```

---
## 🎬 Demo

![MarbleAI in action — real-time fat estimation on meat slices](assets/marbleai.gif)

*The processed video shows YOLO11 segmentation masks overlaid on each meat slice, with real-time fat percentage annotations and a conveyor cutoff line.*

---

## 🏗️ How It Works

### Stage 1 — Instance Segmentation with YOLO11-seg

Each video frame is passed through a fine-tuned YOLO11 segmentation model. Unlike standard object detection (which gives bounding boxes), the segmentation model produces a **pixel-precise mask** for every individual meat slice in the frame. This is what makes per-pixel fat calculation possible.

### Stage 2 — Smart Conveyor Cutoff Logic

This is one of the most thoughtful engineering decisions in the system. A vertical line is drawn across the frame at a configurable ratio (default 35% of frame width). Any meat slice whose centroid has already crossed this line is **skipped entirely** — no segmentation overlay, no fat calculation.

Why? Because a slice that has passed the measurement zone is partially off-screen or partially obscured by adjacent slices. Measuring it at that point would corrupt the fat percentage with incomplete pixel data.

```python
# Skip instances whose centroid has crossed the conveyor cutoff line
if cX >= line_x:
    continue   # No annotation, no calculation — clean skip
```

This is production-aware engineering, not just a research prototype.

### Stage 3 — BGR Color Thresholding + Morphological Cleaning

Inside each segmented slice, fat tissue is identified using BGR (Blue-Green-Red) color thresholds tuned for meat imagery. The thresholds isolate white-to-light-yellow pixels that correspond to intramuscular fat (marbling).

After thresholding, two morphological operations are applied:

- **Morphological Opening** removes small noise pixels that pass the color threshold but are not actually fat
- **Morphological Close** fills small gaps in fat regions to produce a coherent fat mask

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
```

### Stage 4 — Pixel-Level Fat Percentage Calculation

```python
total_area = cv2.countNonZero(seg_mask)    # Total pixels in the slice
fat_area   = cv2.countNonZero(fat_mask)    # Fat pixels inside the slice
fat_pct    = (fat_area / total_area) * 100  # Objective percentage
```

This is not an estimate. It is a pixel count. Every frame, every slice.

### Stage 5 — Annotated Video Output

Fat pixels are painted white on the output frame so the detected marbling is visually obvious. A green contour traces the slice boundary. The fat percentage is overlaid as boxed text at the centroid of each slice.

---

## ⚙️ Full Pipeline

```
Raw Video (.avi)
       │
       ▼
ExtractFrames.py
  Extract individual frames for dataset building and inspection
       │
       ▼
RenameImg.py
  Normalize and rename extracted frames for YOLO annotation
       │
       ▼
 [Manual Annotation]
  Label each meat slice with a segmentation mask in a YOLO-compatible tool
       │
       ▼
TrainingModel.py
  Fine-tune YOLO11-seg on the annotated meat slice dataset
  Output: best.pt (trained segmentation weights)
       │
       ▼
FatCalculation.py  ← Core inference engine
  ┌──────────────────────────────────────────────────────┐
  │  For each frame:                                      │
  │    1. Run YOLO11-seg inference                        │
  │    2. Get pixel masks for each detected slice         │
  │    3. Apply conveyor cutoff line logic                │
  │    4. BGR threshold → fat mask                        │
  │    5. Morphological clean (open + close)              │
  │    6. Compute fat% = fat_pixels / total_pixels        │
  │    7. Annotate frame with contour + fat% label        │
  └──────────────────────────────────────────────────────┘
       │
       ▼
Annotated Output Video (.mp4)
  Each slice tracked, measured, and labeled frame by frame
```

---

## 🔬 Technical Highlights

| Component | Implementation | Why It Matters |
|---|---|---|
| **Segmentation model** | YOLO11s-seg (instance segmentation) | Pixel masks per slice, not just bounding boxes |
| **Color space** | BGR thresholding | Tunable to specific camera and lighting conditions |
| **Noise removal** | Morphological open + close (3×3 ellipse kernel) | Eliminates salt noise and fills fat gaps |
| **Conveyor awareness** | Centroid-based cutoff line | Prevents measurement corruption at frame edges |
| **GPU acceleration** | CUDA device inference | Real-time processing of production-speed conveyors |
| **Output format** | Annotated MP4 at original resolution and FPS | Drop-in review for quality control audits |

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `ExtractFrames.py` | Extract individual frames from raw video footage for dataset creation |
| `RenameImg.py` | Batch rename extracted frames into a normalized format for YOLO annotation |
| `TrainingModel.py` | Fine-tune YOLO11 segmentation on the annotated meat dataset |
| `processing.py` | Lightweight YOLO inference pipeline (raw segmentation visualization, no fat calculation) |
| `FatCalculation.py` | Full production pipeline: segment + threshold + calculate + annotate video |
| `yolo11n.pt` | YOLO11 nano base weights |
| `yolo11s-seg.pt` | YOLO11 small segmentation base weights (used for fine-tuning) |

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/mahmedmajeedai/AUTOMATED-FAT-PERCENTAGE-ESTIMATION-IN-MEAT-SLICES.git
cd AUTOMATED-FAT-PERCENTAGE-ESTIMATION-IN-MEAT-SLICES
```

### 2. Install dependencies

```bash
pip install ultralytics opencv-python numpy
```

### 3. Run fat estimation on your video

Edit the paths in `FatCalculation.py`:

```python
model_path  = "path/to/your/trained/best.pt"
video_path  = "path/to/your/meat/video.avi"
output_path = "path/to/output/annotated.mp4"
```

Run the pipeline:

```bash
python FatCalculation.py
```

### 4. Tune for your setup

```python
# Adjust BGR thresholds for your camera and lighting
lower_white = np.array([74, 120, 212], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)

# Set the conveyor cutoff line position (fraction of frame width)
line_x_ratio = 0.35
```

### 5. Retrain on your own meat data

```bash
# Extract frames from your raw videos
python ExtractFrames.py

# Rename for annotation
python RenameImg.py

# Annotate with a YOLO segmentation tool (e.g. Roboflow, CVAT, LabelImg)

# Retrain
python TrainingModel.py
```

---

## 🏭 Real-World Impact

| Sector | Application |
|---|---|
| 🥩 **Meat Processing Plants** | Automated per-slice fat grading on conveyor belts |
| 🏷️ **Food Labeling Compliance** | Objective, auditable fat content data for nutritional labels |
| 📊 **Quality Control Dashboards** | Per-batch fat distribution reports for production managers |
| 🔬 **Food Science Research** | Quantitative marbling analysis without destructive testing |
| 🛒 **Retail Grading** | Consistent premium vs standard cut classification |

---

## 🔧 Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `imgsz` | `640` | Input image size for YOLO inference |
| `conf` | `0.5` | Minimum confidence threshold for detection |
| `device` | `"cuda"` | Inference device (use `"cpu"` if no GPU available) |
| `lower_white` | `[74, 120, 212]` | Lower BGR bound for fat pixel detection |
| `upper_white` | `[255, 255, 255]` | Upper BGR bound for fat pixel detection |
| `line_x_ratio` | `0.35` | Conveyor cutoff position as fraction of frame width |
| `line_color` | `(255, 255, 0)` | Cutoff line color in BGR (yellow by default) |
| `meat_class_ids` | `None` | Filter by specific YOLO class IDs (None = all classes) |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [Muhammad Ahmed Majeed](https://github.com/mahmedmajeedai)**

*Industrial computer vision for food quality automation*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:c0392b,50:7b1c1c,100:1a0000&height=120&section=footer" width="100%"/>

</div>
