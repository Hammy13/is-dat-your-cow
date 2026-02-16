# Is Dat Your Cow - YOLO-enabled Hybrid ML System

Unique cow identification system using YOLOv11 + Re-ID (7-layer architecture from arch3.puml).

## Architecture Layers

1. **Data Ingestion** - Video stream input & preprocessing
2. **Dataset Preparation** - Frame extraction, quality gates, annotation
3. **Cow Detection** - YOLO detector for bounding boxes
4. **Tracking** - Multi-object tracking (simplified)
5. **Re-ID Feature Extraction** - Appearance encoder for identity signatures
6. **Identity Matching** - Gallery-based cow identification
7. **Outputs & Monitoring** - Dashboard and logs

## Setup Steps

1. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python matplotlib
   ```

2. **Verify YOLO installation**
   ```bash
   yolo version
   ```

3. **Extract frames from video**
   ```bash
   python extract_frames.py
   ```

4. **Train YOLO detector (Layer 3)**
   ```bash
   yolo detect train model=yolo11n.pt data=cow.yaml epochs=10 imgsz=640 batch=8
   ```

5. **Build identity gallery (Layer 6)**
   ```bash
   python build_gallery.py
   ```

6. **Run identification system (Layers 1-7)**
   ```bash
   python cow_identification.py
   ```

7. **View monitoring dashboard (Layer 7)**
   ```bash
   python dashboard.py
   ```

## Project Structure

```
is-dat-your-cow/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── videos/
├── cow.yaml
├── extract_frames.py
└── yolo11n.pt
```

## Training Results

Results will be saved in `runs/detect/train/`
