# Unique Cow Identification System Documentation

## Problem Statement

The objective is to uniquely identify cows across images and videos by combining object detection, pattern-based feature extraction, deep visual embeddings, similarity matching, and temporal tracking. The system is designed as a reproducible research prototype that supports both automated prediction and manual verification.

## Architecture Overview

The implementation follows the `Unique_Cow_Identification_Architecture` and extends it with a persistent gallery and verification workflow:

1. Input images or videos are uploaded into `data/train_uploads` or `data/test_uploads`.
2. YOLO detects cows and extracts bounding box crops.
3. Each crop is processed by:
   - a mesh-grid descriptor branch
   - a deep embedding branch
   - a hybrid fusion branch
4. Training uploads are grouped into `cow_###` identities using clustering.
5. Inference crops are matched against the gallery using cosine similarity.
6. Videos use YOLO tracking / ByteTrack to stabilize identities across frames.
7. Outputs, reports, and gallery artifacts are saved for review.

Architecture source:
- [architecture_unique_cow_poc.puml](/D:/KSAIM/Team_Project/workspace/is-dat-your-cow/cow_identity_prototype/architecture_unique_cow_poc.puml)

## Folder Layout

```text
cow_identity_prototype/
  configs/
    default_config.json
  data/
    train_uploads/
    test_uploads/
      images/
      videos/
  gallery/
    cow_001/
      images/
      embeddings/
      descriptors/
    gallery_index.json
    gallery_metadata.json
  outputs/
    annotated_images/
    annotated_videos/
    fingerprints/
    reports/
    analytics/
    predictions.csv
    match_results.json
    video_tracking_results.csv
    evaluation_summary.json
    similarity_logs.csv
  models/
  src/
  docs/
```

## Data Flow

### Training / Gallery Creation

1. User uploads raw cow images into `data/train_uploads/`.
2. YOLO detects the cow and extracts the crop.
3. Mesh-grid descriptor and deep embedding are computed.
4. Hybrid vectors are clustered using `DBSCAN`.
5. A `cow_###` folder is created for each discovered identity.
6. Crop images, `.npy` embedding files, `.json` descriptor files, and metadata are saved.
7. A global gallery index is written to `gallery/gallery_index.json`.

### Inference

1. User uploads a new image or video to `data/test_uploads/`.
2. YOLO detects cows and extracts crops.
3. Mesh-grid, CNN/ViT, and hybrid vectors are computed.
4. Similarity scores are calculated against gallery identities.
5. If the similarity exceeds the threshold, the matching `Cow_ID` is assigned.
6. If no cow is detected at all, the system rejects the input as `not a cow`.
7. If a cow is detected but no match is strong enough, a new `cow_###` identity is created.
8. Annotated media, reports, and analytics are saved under `outputs/`.

## Algorithms Used

### Detection and Tracking

- Detector: Ultralytics YOLO (`yolo11n.pt` by default)
- Tracker: ByteTrack via YOLO tracking mode

### Mesh-Grid Descriptor

Each cow crop is divided into a regular `N x N` grid. For each cell the system computes:

- dark pixel percentage
- light pixel percentage
- normalized RGB mean
- normalized RGB standard deviation
- hue histogram
- texture statistic
- edge strength

This forms a structured fingerprint that represents the coat pattern distribution.

The implementation also computes a normalized per-grid `grid_score` that reflects how strongly each cell contributes to the mesh descriptor. This score is saved in the mesh metadata and rendered in the fingerprint panel as an annotated grid-score map. In addition, each cell now displays compact per-cell values for grid score (`G`), dark percentage (`D`), light percentage (`L`), texture (`T`), and edge strength (`E`) to improve interpretability for research analysis.

### Deep Embedding Branch

The deep branch extracts a global visual embedding using:

- `ResNet50` by default
- `ViT-B/16` as a configurable option

### Hybrid Matching

The hybrid vector concatenates weighted mesh-grid and deep embedding vectors. Matching is performed with cosine similarity.

For video inference, the Hybrid branch now adds two track-level steps before the identity is treated as stable:

- pose-aware side-view filtering that prefers broad-side, sharper views for identity evidence
- multi-view identity fusion that blends the best views from the same track before final Hybrid scoring

### Gallery Clustering

During the training upload phase, `DBSCAN` groups similar uploaded cows into `cow_###` identities.

Current tuned starting point for this dataset:

- `auto_cluster_eps = 0.24`

This was selected because the original `0.18` setting over-segmented the current upload set into too many singleton clusters.

## Design Decisions

- Gallery-first design instead of full closed-set classification:
  This supports few-shot usage and incremental identity growth.
- Mesh-grid + deep hybrid:
  The mesh branch captures coat layout, while the deep branch captures higher-level visual context.
- Persistent gallery folders:
  This makes manual verification straightforward for research and reporting.
- Streamlit dashboard:
  Chosen for quick inspection and upload-driven experimentation.

## Expected Input Formats

### Training uploads

```text
cow_identity_prototype/data/train_uploads/*.jpg
```

### Test uploads

```text
cow_identity_prototype/data/test_uploads/images/*.jpg
cow_identity_prototype/data/test_uploads/videos/*.mp4
```

### Labeled gallery initialization

```text
cow_identity_prototype/data/train_uploads/cow_001/*.jpg
cow_identity_prototype/data/train_uploads/cow_002/*.jpg
```

## Expected Output Formats

- `gallery/gallery_index.json`: top-level gallery browser index
- `gallery/gallery_metadata.json`: detailed identity metadata
- `outputs/predictions.csv`: image-level or combined prediction export
- `outputs/match_results.json`: match details with top candidates
- `outputs/video_tracking_results.csv`: frame-level video identity records
- `outputs/evaluation_summary.json`: summary statistics
- `outputs/similarity_logs.csv`: similarity score log

## Setup Guide

1. Create or activate the project virtual environment.
2. Install dependencies:

```bash
.venv\Scripts\python.exe -m pip install -r cow_identity_prototype\requirements.txt
```

3. Build the gallery from uploaded images:

```bash
.venv\Scripts\python.exe -m cow_identity_prototype.build_gallery --config cow_identity_prototype\configs\default_config.json
```

4. Run image inference:

```bash
.venv\Scripts\python.exe -m cow_identity_prototype.run_inference --config cow_identity_prototype\configs\default_config.json --mode images --image-dir cow_identity_prototype\data\test_uploads\images
```

5. Run video inference:

```bash
.venv\Scripts\python.exe -m cow_identity_prototype.run_inference --config cow_identity_prototype\configs\default_config.json --mode videos --video-dir cow_identity_prototype\data\test_uploads\videos
```

6. Launch the dashboard:

```bash
.venv\Scripts\python.exe -m streamlit run cow_identity_prototype\streamlit_app.py
```

## Manual Verification Procedure

This is a required part of the system:

1. Run prediction on an image or video.
2. Read the predicted `hybrid_cow_id`.
3. Open the corresponding `gallery/cow_###/images/` folder.
4. Compare the stored gallery images with the test image or video frame.
5. Cross-check the gallery metadata in `gallery/gallery_index.json`.
6. Use the dashboard `Browse Gallery` tab to inspect all images tied to that identity.

For videos:

1. Open the annotated output video.
2. Pick a frame where the predicted `Cow_ID` is clearly visible.
3. Open the gallery folder for that `Cow_ID`.
4. Compare side profile, coat patches, and broad-side pattern visually.

## UI Pages

### Build Gallery

Purpose:
- upload training images
- cluster similar cows
- build `cow_###` folders

Placeholder:
- see `docs/placeholders/build_gallery_placeholder.md`

### Browse Gallery

Purpose:
- select `Cow_ID`
- inspect all gallery images, descriptor files, and vector paths

Placeholder:
- see `docs/placeholders/browse_gallery_placeholder.md`

### Predict Image

Purpose:
- upload a test image
- show annotated output
- display predicted `Cow_ID`
- show matching gallery images side by side

Placeholder:
- see `docs/placeholders/predict_image_placeholder.md`

### Predict Video

Purpose:
- upload a test video
- show annotated video
- inspect tracked Cow_ID assignments
- compare predicted identities with gallery images

Placeholder:
- see `docs/placeholders/predict_video_placeholder.md`

## Evaluation Approach

Recommended evaluation procedure:

1. Build the gallery from known training cows.
2. Hold out a separate set of test images/videos.
3. Compare:
   - YOLO11-only identity branch
   - CNN-only identity branch
   - Hybrid branch
4. Report:
   - number of unique cows detected
   - similarity score distributions
   - detection counts per video
   - identity counts per video
   - manual verification observations

## Limitations

- Side-view scoring is still heuristic and not learned from labeled pose annotations.
- YOLO detections may fail under severe occlusion or mud.
- Non-cow rejection depends on the detector finding zero cow candidates; confident false-positive cow detections can still pass into identity matching.
- Gallery updates during inference can introduce drift if thresholds are too loose.
- The current prototype is optimized for research exploration, not production deployment.

## Future Improvements

- stronger learned re-identification model fine-tuned on cow identities
- stronger track-level aggregation before gallery updates
- SQLite-backed metadata store for larger galleries
- explicit top-k verification panel in the dashboard with similarity bars
