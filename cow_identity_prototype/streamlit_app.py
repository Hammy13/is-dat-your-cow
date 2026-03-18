from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
import plotly.express as px
import streamlit as st

from cow_identity_prototype.config import load_config
from cow_identity_prototype.pipeline import CowIdentityPipeline


st.set_page_config(page_title="Cow Identity Prototype", layout="wide")
st.title("Unique Cow Identification Prototype")

config = load_config()
pipeline = CowIdentityPipeline(config)

st.session_state.setdefault("ui_logs", [])


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


LOG_STYLES = {
    "info": {"border": "#3b82f6", "bg": "#eff6ff", "text": "#1d4ed8", "label": "INFO"},
    "success": {"border": "#16a34a", "bg": "#f0fdf4", "text": "#166534", "label": "SUCCESS"},
    "warning": {"border": "#d97706", "bg": "#fffbeb", "text": "#92400e", "label": "WARNING"},
    "error": {"border": "#dc2626", "bg": "#fef2f2", "text": "#991b1b", "label": "ERROR"},
    "action": {"border": "#7c3aed", "bg": "#f5f3ff", "text": "#5b21b6", "label": "ACTION"},
}

LOG_GROUPS = {
    "gallery": {"title": "Gallery", "icon": "[G]"},
    "image": {"title": "Image", "icon": "[I]"},
    "video": {"title": "Video", "icon": "[V]"},
    "system": {"title": "System", "icon": "[S]"},
}


def _log(message: str, level: str = "info", group: str = "system") -> None:
    style = LOG_STYLES.get(level, LOG_STYLES["info"])
    meta = LOG_GROUPS.get(group, LOG_GROUPS["system"])
    st.session_state["ui_logs"].insert(
        0,
        {
            "timestamp": _timestamp(),
            "message": message,
            "level": level,
            "style": style,
            "group": group,
            "group_title": meta["title"],
            "icon": meta["icon"],
        },
    )
    st.session_state["ui_logs"] = st.session_state["ui_logs"][:60]


def _save_uploaded_files(uploaded_files, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for uploaded in uploaded_files:
        target = target_dir / uploaded.name
        target.write_bytes(uploaded.read())
        saved.append(target)
    return saved


def _load_gallery_index() -> dict:
    index_path = Path(config.paths.gallery_index_json)
    if not index_path.exists():
        return {"cows": {}}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    cows = payload.get("cows", {})
    filtered = {}
    for cow_id, cow_payload in cows.items():
        folder = Path(cow_payload.get("gallery_folder", ""))
        if folder.exists():
            filtered[cow_id] = cow_payload
    payload["cows"] = filtered
    return payload


def _render_gallery_images(paths: list[str], caption_prefix: str) -> None:
    existing_paths = [path for path in paths if Path(path).exists()]
    if not existing_paths:
        st.info("No gallery images available for this Cow_ID yet.")
        return
    columns = st.columns(min(4, max(1, len(existing_paths))))
    for column, image_path in zip(columns, existing_paths[:4]):
        column.image(image_path, caption=f"{caption_prefix}: {Path(image_path).name}", use_container_width=True)


def _render_rescue_evidence(record: pd.Series) -> None:
    rescue_reason = record.get("hybrid_rescue_reason", "")
    rescue_image = record.get("hybrid_rescue_gallery_image", "")
    if not rescue_reason:
        return
    st.markdown("**Rescue Evidence**")
    if rescue_reason == "duplicate_crop_rescue":
        st.info(
            f"Existing ID recovered by direct crop similarity. "
            f"Rescue score: {float(record.get('hybrid_rescue_score', 0.0)):.3f}"
        )
    elif rescue_reason == "local_pattern_rescue":
        st.info(
            f"Existing ID recovered by local coat-pattern alignment. "
            f"Rescue score: {float(record.get('hybrid_rescue_score', 0.0)):.3f}"
        )
    else:
        st.info(f"Existing ID recovered by `{rescue_reason}`.")
    if rescue_image and Path(rescue_image).exists():
        st.markdown(f"**Rescue gallery image:** `{rescue_image}`")
        cols = st.columns(2)
        if record.get("comparison_panel_path") and Path(str(record["comparison_panel_path"])).exists():
            cols[0].image(str(record["comparison_panel_path"]), caption="Query vs gallery comparison", use_container_width=True)
        cols[1].image(str(rescue_image), caption=f"Gallery image used for {rescue_reason}", use_container_width=True)


def _render_log_panel() -> None:
    logs = st.session_state.get("ui_logs", [])
    if not logs:
        logs = [
            {
                "timestamp": _timestamp(),
                "message": "No operations yet.",
                "level": "info",
                "style": LOG_STYLES["info"],
                "group": "system",
                "group_title": "System",
                "icon": "[S]",
            }
        ]
    body = ""
    current_group = None
    for entry in logs:
        style = entry["style"]
        if entry["group"] != current_group:
            current_group = entry["group"]
            body += (
                "<div style='margin:12px 0 8px; font-size:12px; font-weight:700; color:#374151; "
                "text-transform:uppercase; letter-spacing:0.04em;'>"
                f"{entry['icon']} {entry['group_title']}"
                "</div>"
            )
        body += (
            "<div style='margin-bottom:10px; border-left:4px solid {border}; "
            "background:{bg}; color:{text}; padding:10px 12px; border-radius:6px;'>"
            "<div style='font-size:11px; font-weight:700; letter-spacing:0.03em;'>{label} | {timestamp}</div>"
            "<div style='margin-top:4px; font-size:13px; line-height:1.35;'>{message}</div>"
            "</div>"
        ).format(
            border=style["border"],
            bg=style["bg"],
            text=style["text"],
            label=style["label"],
            timestamp=entry["timestamp"],
            message=entry["message"],
        )
    st.markdown(
        (
            "<div style='border-left:2px solid #999; padding-left:14px; min-height: 80vh;'>"
            "<h4 style='margin-top:0'>Operation Log</h4>"
            f"{body}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _list_training_files() -> list[Path]:
    root = Path(config.paths.train_uploads_dir)
    return sorted(path for path in root.rglob("*") if path.is_file())


def _delete_gallery_image(image_path: str) -> int:
    target = Path(image_path)
    if not target.exists():
        return 0
    deleted = 0
    stem = target.stem
    parent = target.parent.parent
    for candidate in [
        target,
        parent / "embeddings" / f"{stem}_cnn.npy",
        parent / "embeddings" / f"{stem}_hybrid.npy",
        parent / "descriptors" / f"{stem}_mesh.npy",
        parent / "descriptors" / f"{stem}_mesh.json",
    ]:
        if candidate.exists():
            candidate.unlink()
            deleted += 1
    return deleted


def _delete_gallery_entry(cow_payload: dict, image_path: str) -> int:
    removed = _delete_gallery_image(image_path)
    gallery_images = cow_payload.get("gallery_images", [])
    source_images = cow_payload.get("source_images", [])
    try:
        index = gallery_images.index(image_path)
    except ValueError:
        index = -1
    if 0 <= index < len(source_images):
        source_path = Path(source_images[index])
        if source_path.exists():
            source_path.unlink()
            removed += 1
    return removed


def _refresh_gallery_from_disk() -> None:
    path = pipeline.build_gallery_from_uploads(config.paths.train_uploads_dir, config.paths.gallery_json)
    _log(f"Refreshed gallery metadata and clustering after deletion. Output: {path}")


main_col, log_col = st.columns([4.2, 1.3])

with main_col:
    gallery_tab, browser_tab, image_tab, video_tab, analytics_tab, docs_tab = st.tabs(
        ["Build Gallery", "Browse Gallery", "Predict Image", "Predict Video", "Analytics", "Docs"]
    )

    with gallery_tab:
        st.subheader("Training / Gallery Creation")
        uploaded_training = st.file_uploader(
            "Upload training cow images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="training_uploads",
        )
        if st.button("Save Training Uploads"):
            if not uploaded_training:
                st.warning("Upload one or more training images first.")
                _log("Attempted to save training uploads with no files selected.", "warning", "gallery")
            else:
                saved = _save_uploaded_files(uploaded_training, Path(config.paths.train_uploads_dir))
                st.success(f"Saved {len(saved)} training images to {config.paths.train_uploads_dir}")
                _log(f"Saved {len(saved)} training images to train_uploads.", "success", "gallery")

        training_files = _list_training_files()
        selected_training_files = st.multiselect(
            "Remove specific training data",
            options=[str(path.relative_to(Path(config.paths.train_uploads_dir))) for path in training_files],
            help="Select one or more uploaded training files to delete from train_uploads.",
        )
        if st.button("Delete Selected Training Files"):
            if not selected_training_files:
                st.warning("Select at least one training file to delete.")
                _log("Attempted to delete training data with no files selected.", "warning", "gallery")
            else:
                deleted = 0
                for relative_name in selected_training_files:
                    target = Path(config.paths.train_uploads_dir) / relative_name
                    if target.exists():
                        target.unlink()
                        deleted += 1
                st.success(f"Deleted {deleted} training files. Rebuild the gallery to apply the change.")
                _log(f"Deleted {deleted} training files from train_uploads.", "action", "gallery")

        col1, col2 = st.columns(2)
        if col1.button("Build Gallery From Uploads"):
            path = pipeline.build_gallery_from_uploads(config.paths.train_uploads_dir, config.paths.gallery_json)
            st.success(f"Gallery built at {path}")
            _log(f"Built gallery from uploads. Output: {path}", "success", "gallery")
        if col2.button("Initialize Gallery From Labeled Folders"):
            path = pipeline.initialize_gallery(config.paths.gallery_dataset_dir, config.paths.gallery_json)
            st.success(f"Labeled gallery built at {path}")
            _log(f"Initialized gallery from labeled folders. Output: {path}", "success", "gallery")
        st.caption("Auto-clustering uses YOLO crops + mesh-grid + deep embeddings, then groups similar cows into `gallery/cow_###`.")

    with browser_tab:
        st.subheader("Manual Verification Browser")
        gallery_index = _load_gallery_index()
        cow_ids = sorted(gallery_index.get("cows", {}).keys())
        if not cow_ids:
            st.info("Build the gallery first. Then Cow_ID folders and metadata will appear here.")
        else:
            browser_key = "browse_gallery_selected_cow_id"
            if st.session_state.get(browser_key) not in cow_ids:
                st.session_state[browser_key] = cow_ids[0]
            selected_cow_id = st.selectbox("Cow_ID", cow_ids, key=browser_key)
            cow_payload = gallery_index["cows"][selected_cow_id]
            st.write(f"Folder: `{cow_payload['gallery_folder']}`")
            st.write(f"Observation count: `{cow_payload['observation_count']}`")
            gallery_images = [path for path in cow_payload.get("gallery_images", []) if Path(path).exists()]
            _render_gallery_images(gallery_images, selected_cow_id)
            selected_gallery_images = st.multiselect(
                "Delete specific gallery images for this Cow_ID",
                options=gallery_images,
                format_func=lambda path: Path(path).name,
                key=f"gallery_delete_{selected_cow_id}",
            )
            col_a, col_b = st.columns(2)
            if col_a.button("Delete Selected Gallery Images", key=f"delete_images_{selected_cow_id}"):
                if not selected_gallery_images:
                    st.warning("Select at least one gallery image to delete.")
                    _log(f"Attempted to delete gallery images from {selected_cow_id} with no selection.", "warning", "gallery")
                else:
                    removed_files = 0
                    for image_path in selected_gallery_images:
                        removed_files += _delete_gallery_entry(cow_payload, image_path)
                    st.success(f"Deleted {len(selected_gallery_images)} gallery image(s) and linked source data. Refreshing gallery.")
                    _log(
                        f"Deleted {len(selected_gallery_images)} gallery image(s) from {selected_cow_id}; removed {removed_files} files including linked sources.",
                        "action",
                        "gallery",
                    )
                    _refresh_gallery_from_disk()
                    st.rerun()
            if col_b.button("Delete Entire Cow_ID", key=f"delete_cow_{selected_cow_id}"):
                folder = Path(cow_payload["gallery_folder"])
                for source_image in cow_payload.get("source_images", []):
                    source_path = Path(source_image)
                    if source_path.exists():
                        source_path.unlink()
                if folder.exists():
                    shutil.rmtree(folder)
                    st.success(f"Deleted {selected_cow_id} from gallery and linked source uploads. Refreshing gallery.")
                    _log(f"Deleted entire gallery folder for {selected_cow_id} and linked source uploads.", "action", "gallery")
                    _refresh_gallery_from_disk()
                    if browser_key in st.session_state:
                        del st.session_state[browser_key]
                    st.rerun()
                else:
                    st.warning("Selected gallery folder no longer exists.")
                    _log(f"Selected gallery folder for {selected_cow_id} no longer exists; refreshing gallery.", "warning", "gallery")
                    _refresh_gallery_from_disk()
                    if browser_key in st.session_state:
                        del st.session_state[browser_key]
                    st.rerun()
            if st.button("Rebuild / Refresh Gallery Metadata", key=f"refresh_gallery_{selected_cow_id}"):
                _refresh_gallery_from_disk()
                st.success("Gallery metadata rebuilt from current train_uploads.")
                _log(f"Rebuilt gallery metadata from train_uploads for browser refresh.", "success", "gallery")
                st.rerun()
            st.json(cow_payload)

    with image_tab:
        st.subheader("Image Inference")
        update_gallery_image = st.checkbox("Update gallery with inference results", value=False, key="image_update_gallery")
        uploaded_image = st.file_uploader("Upload test cow image", type=["jpg", "jpeg", "png"], key="image_upload")
        if uploaded_image is not None:
            saved = _save_uploaded_files([uploaded_image], Path(config.paths.test_image_dir))
            _log(
                f"Running image inference for {saved[0].name}. Gallery update is {'ON' if update_gallery_image else 'OFF'}.",
                "action",
                "image",
            )
            result = pipeline.infer_images(saved[0], config.paths.gallery_json, "streamlit_image", update_gallery=update_gallery_image)
            records = pd.DataFrame(result["records"])
            annotated_path = Path(config.paths.output_root) / "annotated_images" / f"{saved[0].stem}_annotated.png"
            left, right = st.columns([1.1, 0.9])
            if annotated_path.exists():
                left.image(str(annotated_path), caption="Annotated query image", use_container_width=True)
            right.dataframe(records, use_container_width=True)
            if not records.empty:
                best = records.iloc[0]
                st.markdown(f"**Predicted Hybrid Cow_ID:** `{best['hybrid_cow_id']}`")
                if bool(best["hybrid_is_new_identity"]):
                    st.warning("This query was treated as a new cow. The crop has been saved into a new gallery folder.")
                elif best.get("hybrid_rescue_reason"):
                    st.info(f"Existing identity was recovered by `{best['hybrid_rescue_reason']}`.")
                st.markdown(f"**Manual verification folder:** `{best['hybrid_gallery_folder']}`")
                if best.get("saved_gallery_image_path"):
                    st.markdown(f"**Saved gallery crop:** `{best['saved_gallery_image_path']}`")
                if best.get("comparison_panel_path") and Path(best["comparison_panel_path"]).exists():
                    st.image(best["comparison_panel_path"], caption="Query vs matched gallery comparison", use_container_width=True)
                _render_rescue_evidence(best)
                st.markdown("**Gallery images for manual verification**")
                _render_gallery_images(best["hybrid_gallery_images"], best["hybrid_cow_id"])
                st.markdown("**Conclusion**")
                st.success(best["conclusion"])
                _log(
                    f"Image inference complete for {saved[0].name}: {best['conclusion']}",
                    "success" if not best["hybrid_is_new_identity"] else "warning",
                    "image",
                )

    with video_tab:
        st.subheader("Video Inference")
        update_gallery_video = st.checkbox("Update gallery with inference results", value=False, key="video_update_gallery")
        uploaded_video = st.file_uploader("Upload test cow video", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
        if uploaded_video is not None:
            saved = _save_uploaded_files([uploaded_video], Path(config.paths.test_video_dir))
            _log(
                f"Running video inference for {saved[0].name}. Gallery update is {'ON' if update_gallery_video else 'OFF'}.",
                "action",
                "video",
            )
            result = pipeline.infer_videos(saved[0], config.paths.gallery_json, "streamlit_video", update_gallery=update_gallery_video)
            annotated_path = Path(config.paths.output_root) / "annotated_videos" / f"{saved[0].stem}_annotated.mp4"
            if annotated_path.exists():
                st.video(str(annotated_path))
            records = pd.DataFrame(result["records"])
            st.dataframe(records.head(200), use_container_width=True)
            if not records.empty:
                summary = records.groupby("hybrid_cow_id").size().reset_index(name="frames_seen")
                st.dataframe(summary, use_container_width=True)
                selected_cow_id = st.selectbox("Show gallery images for predicted Cow_ID", summary["hybrid_cow_id"].tolist(), key="video_cow_picker")
                gallery_index = _load_gallery_index()
                payload = gallery_index.get("cows", {}).get(selected_cow_id, {})
                matching_rows = records[records["hybrid_cow_id"] == selected_cow_id]
                if not matching_rows.empty:
                    best_row = matching_rows.iloc[0]
                    if best_row.get("comparison_panel_path") and Path(best_row["comparison_panel_path"]).exists():
                        st.image(best_row["comparison_panel_path"], caption=f"Comparison panel for {selected_cow_id}", use_container_width=True)
                    _render_rescue_evidence(best_row)
                    st.markdown("**Conclusion**")
                    st.success(best_row["conclusion"])
                    _log(
                        f"Video inference complete for {saved[0].name}: {best_row['conclusion']}",
                        "success" if not best_row["hybrid_is_new_identity"] else "warning",
                        "video",
                    )
                _render_gallery_images(payload.get("gallery_images", []), selected_cow_id)

    with analytics_tab:
        st.subheader("Analytics")
        reports_dir = Path(config.paths.output_root) / "reports"
        report_files = sorted(reports_dir.glob("*.json"))
        chart_htmls = sorted((Path(config.paths.output_root) / "analytics").glob("*.html"))
        if not report_files:
            st.info("Run gallery building or inference first.")
        else:
            selected = st.selectbox("Report file", report_files, format_func=lambda path: path.name)
            payload = json.loads(selected.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                dataframe = pd.DataFrame(payload)
                st.dataframe(dataframe, use_container_width=True)
                if not dataframe.empty and "hybrid_score" in dataframe.columns:
                    fig = px.histogram(dataframe, x="hybrid_score", nbins=20, title="Hybrid Similarity Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    comparison = pd.DataFrame(
                        {
                            "model": ["YOLO11", "CNN", "Hybrid"],
                            "unique_ids": [
                                dataframe["yolo11_cow_id"].nunique(),
                                dataframe["cnn_cow_id"].nunique(),
                                dataframe["hybrid_cow_id"].nunique(),
                            ],
                        }
                    )
                    st.plotly_chart(px.bar(comparison, x="model", y="unique_ids", title="YOLO11 vs CNN vs Hybrid"), use_container_width=True)
        if chart_htmls:
            st.caption("Open the saved Plotly HTML charts from the outputs/analytics folder for full interactive browsing.")

    with docs_tab:
        st.subheader("Documentation")
        doc_path = Path(config.paths.docs_root) / "SYSTEM_DOCUMENTATION.md"
        if doc_path.exists():
            st.markdown(doc_path.read_text(encoding="utf-8"))
        else:
            st.info("Documentation file not found.")

with log_col:
    _render_log_panel()
