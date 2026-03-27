# Cow Identity Prototype

This is a research-oriented proof-of-concept for unique cow identification across images and videos using:

- Ultralytics YOLO detection and tracking
- mesh-grid coat pattern descriptors
- deep feature embeddings from `ResNet50` or `ViT`
- hybrid similarity matching against a persistent gallery
- manual verification support through gallery browsing

## Project Layout

```text
cow_identity_prototype/
  configs/
  data/
    train_uploads/
    labeled_gallery_train/
    test_uploads/
      images/
      videos/
  docs/
  gallery/
    cow_001/
    cow_002/
    gallery_index.json
    gallery_metadata.json
  models/
  outputs/
    annotated_images/
    annotated_videos/
    analytics/
    reports/
    fingerprints/
    predictions.csv
    match_results.json
    video_tracking_results.csv
    evaluation_summary.json
  src/
```

## Main Workflows

### 1. Build gallery from unlabeled uploaded images

```bash
.venv\Scripts\python.exe -m cow_identity_prototype.build_gallery --config cow_identity_prototype\configs\default_config.json
```

This uses YOLO crops, mesh-grid descriptors, deep embeddings, and DBSCAN clustering to group visually similar cows into `gallery/cow_###`.

### 2. Initialize gallery from labeled folders

```bash
.venv\Scripts\python.exe -m cow_identity_prototype.initialize_gallery --config cow_identity_prototype\configs\default_config.json
```

Create your manual training folders here before running that command:

```text
cow_identity_prototype/data/labeled_gallery_train/COW_001/
cow_identity_prototype/data/labeled_gallery_train/COW_002/
```

Each folder name is used directly as the gallery `Cow_ID`.

You can also do this from the Streamlit UI:

1. Open the `Build Gallery` tab.
2. Enter a new folder name such as `COW_001`.
3. Click `Create Cow Folder`.
4. Upload one or more images into that folder with `Save Images To Cow Folder`.
5. Click `Build Gallery From Manual Cow_ID Folders`.

### 3. Run image inference

```bash
.venv\Scripts\python.exe -m cow_identity_prototype.run_inference --config cow_identity_prototype\configs\default_config.json --mode images --image-dir path\to\query.jpg --output-prefix image_demo
```

### 4. Run video inference

```bash
.venv\Scripts\python.exe -m cow_identity_prototype.run_inference --config cow_identity_prototype\configs\default_config.json --mode videos --video-dir path\to\query.mp4 --output-prefix video_demo
```

### 5. Launch dashboard

```bash
.venv\Scripts\python.exe -m streamlit run cow_identity_prototype\streamlit_app.py
```

## Manual Verification

After prediction:

1. Note the predicted `hybrid_cow_id`.
2. Open [gallery](/D:/KSAIM/Team_Project/workspace/is-dat-your-cow/cow_identity_prototype/gallery).
3. Open the matching folder such as `gallery/cow_001/`.
4. Compare the stored gallery crop images with the current query image or video frame.
5. Check [gallery_index.json](/D:/KSAIM/Team_Project/workspace/is-dat-your-cow/cow_identity_prototype/gallery/gallery_index.json) for the full list of images, descriptors, and vectors tied to that `Cow_ID`.

## Documentation

The full research-style write-up is in [SYSTEM_DOCUMENTATION.md](/D:/KSAIM/Team_Project/workspace/is-dat-your-cow/cow_identity_prototype/docs/SYSTEM_DOCUMENTATION.md).
