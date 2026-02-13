# Is Dat Your Cow - YOLO Object Detection

Cow detection system using YOLOv11.

## Setup Steps

1. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python
   ```

2. **Verify YOLO installation**
   ```bash
   yolo version
   ```

3. **Extract frames from video**
   ```bash
   python extract_frames.py
   ```

4. **Train the model**
   ```bash
   yolo detect train model=yolo11n.pt data=cow.yaml epochs=10 imgsz=640 batch=8
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
