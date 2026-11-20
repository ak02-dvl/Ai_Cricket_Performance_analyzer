"""Standalone Streamlit human-labeling UI; does NOT import or change CricketAnalyzer."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shot_dataset_v40 import (  # noqa: E402
    ALL_LABELS, HAND, REVIEW, VIEWS, load_csv, validate_manifest, write_csv,
)

st.set_page_config(page_title="Cricket Shot Ground Truth v4.0A", page_icon="🏏", layout="wide")
st.title("🏏 Shot Classification v4.0A — Human Ground Truth")
st.caption("Label original video only. Do not copy the experimental ShotTypeV2 guess as ground truth.")

manifest_path = Path(st.sidebar.text_input("Manifest CSV", "data/shot_labels_v40.csv"))
clip_dir = Path(st.sidebar.text_input("Extracted clips folder", "data/shot_review_v40"))
if not manifest_path.is_file():
    st.error(f"Manifest not found: {manifest_path}")
    st.stop()

try:
    rows = load_csv(manifest_path)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

status = validate_manifest(rows)
st.metric("Clips", len(rows))
st.metric("Human-reviewed", status["reviewed_clips"])
if status["errors"]:
    st.error("Manifest has validation errors: " + "; ".join(status["errors"]))
if not rows:
    st.warning("No clips found. Generate an initial manifest from the JSON reports.")
    st.stop()

clip_id = st.selectbox("Select an impact-centered clip", [r["clip_id"] for r in rows])
row = next(r for r in rows if r["clip_id"] == clip_id)
clip = clip_dir / (clip_id + ".mp4")
if clip.is_file():
    st.video(str(clip))
else:
    st.warning(f"Review clip missing: {clip}. Extract the original video first.")

st.write("Source:", row["source_video"])
st.write("Impact frame (estimated):", row["impact_frame"])
st.write("Clip frame range:", f"{row['start_frame']}–{row['end_frame']}")

with st.form("shot_review_form"):
    def index_of(seq, value):
        return seq.index(value) if value in seq else 0

    label = st.selectbox("Shot label", ALL_LABELS, index=index_of(ALL_LABELS, row["shot_label"]))
    view = st.selectbox("Camera view", VIEWS, index=index_of(VIEWS, row["camera_view"]))
    hand = st.selectbox("Batting stance", HAND, index=index_of(HAND, row["batting_stance"]))
    annotator = st.text_input("Reviewer / annotator", row["annotator"])
    notes = st.text_area("Evidence and uncertainty notes", row["notes"])
    watched = st.checkbox("I inspected the original clip, not just the detector output")
    save = st.form_submit_button("Save human review")

if save:
    if not watched or not clip.is_file():
        st.error("Inspect the original clip first; it must exist on disk.")
    elif label == "Unlabeled":
        st.error("Choose a reviewed class, Other, or Ambiguous. Leave Unlabeled clips unreviewed.")
    elif not annotator.strip():
        st.error("Enter reviewer name or identifier.")
    else:
        updated = [dict(item) for item in rows]
        target = next(item for item in updated if item["clip_id"] == clip_id)
        target.update({"shot_label": label, "review_status": "Reviewed",
                       "camera_view": view, "batting_stance": hand,
                       "annotator": annotator.strip(), "notes": notes,
                       "split": ""})  # never silently assign a train/test split
        checked = validate_manifest(updated)
        if checked["errors"]:
            st.error("Validation errors: " + "; ".join(checked["errors"]))
        else:
            write_csv(manifest_path, updated)
            st.success(f"Saved human review for {clip_id}. No model was trained.")
            st.info("Reload the page to see the saved label; set train/val/test by recording later.")

st.info("This UI records human annotations only. The v4.0 classifier remains untrained until a labeled, diverse dataset is available.")
