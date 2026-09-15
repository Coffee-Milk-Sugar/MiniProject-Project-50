# Traffic Sign Detection, Segmentation & Classification Pipeline

An end-to-end computer vision pipeline combining **YOLO**, **SAM 2.1**, and **ResNet-18** to detect, segment, and classify traffic signs in test images with zero-setup automatic weight downloading and dependency management.

---

## ⚡ CUDA Acceleration & Performance

For optimal real-time demonstration and meeting strict performance targets, running this pipeline on an **NVIDIA GPU with CUDA enabled** is strongly recommended.

| Processing Mode | Average Processing Time per Image | Pipeline Bottleneck |
| --- | --- | --- |
| **CUDA (GPU Accelerated)** | **< 2.0 seconds** (~0.8s – 1.6s) | Real-time GPU parallel inference |
| **CPU Only** | ~8.0 – 15.0 seconds | Heavy SAM 2.1 attention matrix computations |

* **Why CUDA Matters:** SAM 2.1 (Segment Anything Model) relies on transformer-based vision encoders. Enabling CUDA allows tensor operations for YOLO detection, SAM prompt masking, and ResNet-18 classification to run concurrently on GPU cores, driving total processing time down **below the 2-second target per image**.
* **Verification:** The script automatically detects CUDA via `torch.cuda.is_available()` and routes tensor computations to `cuda` if available, falling back to `cpu` otherwise.

---

## 🔑 Key Features

* **Zero Manual Dependency Setup:** Features a built-in auto-installer that detects and installs missing packages (`ultralytics`, `torch`, `opencv-python`, etc.) via `pip` on launch.
* **Automated Weight Retrieval:** Automatically downloads missing model weights directly from GitHub Releases (`custom_model_weights`) with progress bars on the first run.
* **3-Stage Hybrid Deep Learning Pipeline:**
1. **YOLO (Detection):** Identifies region-of-interest (ROI) bounding boxes with confidence filtering.
2. **SAM 2.1 (Segmentation):** Refines bounding boxes into tight object masks using morphological closing, connected-component analysis, and convex hull polygon fitting.
3. **ResNet-18 (Classification):** Classifies transparent BGRA-cropped sign segments across 48 traffic sign classes.


* **Interactive 3-Panel Visualizer:** Displays side-by-side comparison panels (*Original Image* | *SAM Mask* | *Segmented Result + Class Label*) normalized for high-density caption rendering.

---

## 📁 Repository Structure

```text
├── InputTest/                    # Input folder for test images (.png, .jpg, .jpeg, .ppm)
├── Outputs/                      # Auto-generated output directories
│   ├── Annotated_Detections/     # SAM 2.1 mask overlays and detection boxes
│   ├── Final_Results/            # Side-by-side 3-panel visualization panels
│   └── ShapedCrops_SAM2/         # Exported 4-channel BGRA transparent sign crops
├── inputFiles.txt                 # Optional text file containing test file listings
├── main.py                       # Core executable script
├── requirements.txt              # Standard package requirements file
└── README.md                     # Project documentation

```

---

## 🛠️ Installation & Execution

### 1. Prerequisites

Ensure **Python 3.8+** and **CUDA Toolkit** (for GPU acceleration) are installed. Verify PyTorch CUDA recognition:

```bash
python -c "import torch; print(torch.cuda.is_available())"

```

*(Should return `True` for GPU acceleration)*

### 2. Run the Pipeline

Simply execute the main script. All dependencies and weights (`best.pt`, `sam2.1_b.pt`, `resnet18_traffic_sign.pth`) will download automatically if missing:

```bash
python main.py

```

---

## 🎮 Interactive Navigation Controls

When running the presentation window:

* **ANY KEY / SPACE:** Step forward to process and view the next image.
* **ESC:** Exit the application cleanly.
