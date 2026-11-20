"""Streamlit frontend for the AI Cricket Performance Analyzer.

The detection/phase/impact logic stays inside pipeline.analyzer.CricketAnalyzer.
This UI preserves all shot-completion, impact-confirmation, validity gating,
and JSON/PDF export semantics while providing a modern sports-analytics UI.
"""
import csv
import html
import json
import math
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from copy import deepcopy

import base64
import cv2
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

from pipeline.analyzer import CricketAnalyzer
from analysis.score_evidence import build_score_evidence
from analysis.shot_type_v2 import ShotTypeV2
from analysis.impact_centered_v3 import ImpactCenteredMotionV3
from analysis.bat_ball_motion_v32 import BatBallMotionV32
from analysis.detection_evidence_v33 import DetectionEvidenceV33
from analysis.tracker_association_v34 import TrackerAssociationV34
from pdf_generator import generate_pdf_report


# ============================================================
# DESIGN TOKENS — DARK SPORTS-TECH PALETTE
# ============================================================

PALETTE = {
    "bg_main": "#0B1120",
    "bg_sidebar": "#111827",
    "bg_card": "#172338",
    "bg_card_hover": "#1E2E4A",
    "blue_primary": "#3B82F6",
    "blue_secondary": "#60A5FA",
    "text_main": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "success": "#22C55E",
    "error": "#EF4444",
    "warning": "#F59E0B",
    "border": "rgba(96, 165, 250, 0.18)",
}

NAV_SECTIONS = [
    ("Dashboard", "📊"),
    ("Upload Video", "📤"),
    ("Video Analysis", "🎬"),
    ("Performance Results", "🏆"),
    ("Technique Analysis", "📐"),
    ("Footwork Analysis", "👟"),
    ("Shot Analysis", "🏏"),
    ("Reports and Downloads", "📑"),
]


# ============================================================
# SAFE HELPERS (PRESERVED FROM ORIGINAL APP.PY)
# ============================================================

def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def is_finite_number(value):
    try:
        if isinstance(value, bool) or value is None:
            return False
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def is_completed_shot(phase, shot):
    return (
        str(phase).strip().lower() in ("follow through", "recovery")
        and str(shot).strip().lower() == "shot played"
    )


class ShotCompletionGate:
    """Mirror the confirmed-impact completion behavior in test_pipeline.py.

    Never stop early based only on a shot/phase classification. Require
    confirmed impact, a stable finished shot, and the original grace window.
    """

    def __init__(self, grace_frames=8, confirmation_frames=2, stable_frames=5):
        self.grace_frames = max(0, int(grace_frames))
        self.confirmation_frames = max(2, int(confirmation_frames))
        self.stable_frames = max(1, int(stable_frames))
        self.last_swing = None
        self.completion_start = None
        self.stable_count = 0

    def reset_completion(self):
        self.completion_start = None
        self.stable_count = 0

    def update(self, frame_number, phase, shot, impact_confirmed=False):
        phase = str(phase).strip().lower()
        shot = str(shot).strip().lower()

        if phase in ("downswing", "impact"):
            self.last_swing = frame_number

        if not impact_confirmed:
            self.reset_completion()
            return False

        completed = phase in ("follow through", "recovery") and shot == "shot played"
        if not completed:
            self.reset_completion()
            return False

        if self.completion_start is None:
            self.completion_start = frame_number
        self.stable_count += 1

        anchor = self.last_swing if self.last_swing is not None else self.completion_start
        deadline = anchor + self.grace_frames + self.confirmation_frames + 1
        return self.stable_count >= self.stable_frames and frame_number >= deadline


def make_impact_summary(report, impact, current_frame):
    """Use the impact-time values saved by CricketAnalyzer where present."""
    estimated_frame = impact.get("impact_frame") or report.get("impact_frame") or current_frame
    return {
        "impact_frame": estimated_frame,
        "confirmation_frame": impact.get("confirmation_frame") or current_frame,
        "phase": report.get("impact_phase") or report.get("phase", "Impact"),
        "shot": report.get("impact_shot") or report.get("shot", "Unknown"),
        "score": safe_float(report.get("impact_score", report.get("overall_score", 0))),
        "evidence_status": impact.get("status", "Impact estimated"),
    }


def json_serializable(value):
    """Convert common NumPy/model scalar values if present in report data."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return str(value)


def resize_frame_for_display(frame, max_width=860, max_height=540):
    """
    Resize only the UI display copy.
    Preserve the original aspect ratio and never enlarge the frame.
    """
    if frame is None or frame.size == 0:
        return frame

    frame_height, frame_width = frame.shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return frame

    scale = min(max_width / frame_width, max_height / frame_height, 1.0)
    if scale >= 1.0:
        return frame

    new_width = max(1, int(round(frame_width * scale)))
    new_height = max(1, int(round(frame_height * scale)))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# SHOT LABELS — V4.0A (PRESERVED)
# ============================================================

HUMAN_SHOT_LABELS = {
    "Drive",
    "Cut",
    "Pull",
    "Defense",
    "Other",
    "Ambiguous",
}


def load_human_shot_annotation(video_name):
    """
    Load a reference shot label for an exact source filename.
    """
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "shot_labels_v40.csv",
    )

    target_name = os.path.basename(str(video_name or "")).strip().casefold()

    result = {
        "status": "not_reviewed",
        "source_video": os.path.basename(str(video_name or "")),
        "shot_type": None,
        "clip_id": None,
        "impact_frame": None,
        "start_frame": None,
        "end_frame": None,
    }

    if not target_name:
        return result

    if not os.path.isfile(manifest_path):
        result["status"] = "manifest_missing"
        return result

    try:
        with open(manifest_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
    except (OSError, csv.Error) as exc:
        result["status"] = "manifest_error"
        result["message"] = str(exc)
        return result

    matches = []
    for row in rows:
        source_video = os.path.basename((row.get("source_video") or "").strip()).casefold()
        review_status = (row.get("review_status") or "").strip()
        shot_label = (row.get("shot_label") or "").strip()

        if (
            source_video == target_name
            and review_status == "Reviewed"
            and shot_label in HUMAN_SHOT_LABELS
        ):
            matches.append(row)

    if len(matches) > 1:
        result["status"] = "multiple_matches"
        return result

    if not matches:
        return result

    row = matches[0]
    return {
        "status": "reviewed",
        "source_video": (row.get("source_video") or "").strip(),
        "shot_type": (row.get("shot_label") or "").strip(),
        "clip_id": (row.get("clip_id") or "").strip(),
        "impact_frame": (row.get("impact_frame") or "").strip(),
        "start_frame": (row.get("start_frame") or "").strip(),
        "end_frame": (row.get("end_frame") or "").strip(),
    }


# ============================================================
# SESSION STATE MANAGEMENT
# ============================================================

def init_session_state():
    defaults = {
        "active_nav": "Dashboard",
        "uploaded_video_meta": None,
        "analysis_bundle": None,
        "recent_analyses": [],
        "trigger_analysis": False,
        "batting_stance": "Unknown",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def navigate_to(section_name):
    st.session_state["active_nav"] = section_name


def inspect_video_metadata(video_path, file_name, file_size_bytes):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0)
        if fps <= 0:
            fps = 30.0
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        duration = total_frames / fps if fps > 0 else 0.0
        return {
            "path": video_path,
            "name": file_name,
            "size_mb": file_size_bytes / (1024 * 1024),
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration,
        }
    finally:
        cap.release()


def download_video_from_url(video_url, temp_dir):
    """Download a video from a direct URL, Google Drive share link, or YouTube/yt-dlp URL."""
    clean_url = video_url.strip()
    parsed = urllib.parse.urlparse(clean_url)
    host = parsed.netloc.lower()
    is_youtube_or_social = any(
        domain in host
        for domain in ("youtube.com", "youtu.be", "instagram.com", "twitter.com", "x.com", "vimeo.com")
    )

    # Convert Google Drive view link to direct download link
    gdrive_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", clean_url)
    if gdrive_match:
        file_id = gdrive_match.group(1)
        clean_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # 1. Use yt-dlp for YouTube / web video platforms
    yt_error = None
    try:
        import yt_dlp
        out_template = os.path.join(temp_dir, "%(title).60s.%(ext)s")
        ydl_opts = {
            "format": "best[vcodec!=none][acodec!=none]/best[vcodec!=none]/best",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "tv_embedded", "web"],
                }
            },
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(clean_url, download=True)
            # Find the actual downloaded/merged video file in temp_dir
            candidates = [
                os.path.join(temp_dir, f)
                for f in os.listdir(temp_dir)
                if os.path.isfile(os.path.join(temp_dir, f)) and not f.endswith(".part")
            ]
            if candidates:
                downloaded_path = max(candidates, key=os.path.getsize)
                safe_name = os.path.basename(downloaded_path)
                return downloaded_path, safe_name, os.path.getsize(downloaded_path)
    except ImportError:
        yt_error = "yt-dlp is not installed. Run: pip install yt-dlp"
    except Exception as exc:
        yt_error = str(exc)

    # Do not fall back to raw HTML download for YouTube/social URLs
    if is_youtube_or_social:
        raise RuntimeError(f"YouTube/web video download failed: {yt_error or 'No playable stream found.'}")

    # 2. Direct HTTP/HTTPS video file download (.mp4, .avi, .mov, Google Drive)
    url_basename = os.path.basename(parsed.path)
    if os.path.splitext(url_basename)[1].lower() not in (".mp4", ".avi", ".mov", ".mkv"):
        url_basename = "linked_batting_video.mp4"

    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", url_basename)
    input_path = os.path.join(temp_dir, safe_name)

    req = urllib.request.Request(
        clean_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        content_type = resp.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError("The URL returned an HTML webpage instead of a direct video file.")
        with open(input_path, "wb") as out_f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)

    return input_path, safe_name, os.path.getsize(input_path)

# ============================================================
# UI STYLING & REUSABLE COMPONENTS
# ============================================================

def inject_custom_css():
    palette = globals().get("PALETTE", {})
    bg_main = palette.get("bg_main", "#0B1120")
    bg_sidebar = palette.get("bg_sidebar", "#0F172A")
    bg_card = palette.get("bg_card", "#172338")
    bg_card_hover = palette.get("bg_card_hover", "#1E293B")
    blue_primary = palette.get("blue_primary", "#3B82F6")
    blue_secondary = palette.get("blue_secondary", "#60A5FA")
    text_main = palette.get("text_main", "#F8FAFC")
    text_secondary = palette.get("text_secondary", "#94A3B8")
    success = palette.get("success", "#22C55E")
    error = palette.get("error", "#EF4444")
    warning = palette.get("warning", "#F59E0B")
    border = palette.get("border", "rgba(148, 163, 184, 0.18)")

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg-main: {bg_main};
            --bg-sidebar: {bg_sidebar};
            --bg-card: {bg_card};
            --bg-card-hover: {bg_card_hover};
            --blue-primary: {blue_primary};
            --blue-secondary: {blue_secondary};
            --text-main: {text_main};
            --text-secondary: {text_secondary};
            --success: {success};
            --error: {error};
            --warning: {warning};
            --border: {border};
        }}

        html, body, .stApp {{
            background-color: var(--bg-main) !important;
            color: var(--text-main) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        /* Preserve Streamlit Material Icons globally */
        [data-testid="stIconMaterial"],
        .material-symbols-rounded {{
            font-family: 'Material Symbols Rounded' !important;
        }}

        /* Dark Top Header & High-Contrast Sidebar Toggle Arrow */
        header[data-testid="stHeader"] {{
            background-color: var(--bg-main) !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12) !important;
        }}

        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] span,
        div[data-testid="stToolbar"] button,
        div[data-testid="stToolbar"] span {{
            color: #F8FAFC !important;
        }}

        button[data-testid="stSidebarCollapseButton"],
        button[data-testid="stExpandSidebarButton"],
        div[data-testid="stSidebarCollapsedControl"] button {{
            background-color: rgba(59, 130, 246, 0.22) !important;
            border: 1px solid rgba(96, 165, 250, 0.55) !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
            opacity: 1 !important;
        }}

        button[data-testid="stSidebarCollapseButton"]:hover,
        button[data-testid="stExpandSidebarButton"]:hover,
        div[data-testid="stSidebarCollapsedControl"] button:hover {{
            background-color: rgba(59, 130, 246, 0.42) !important;
            border-color: #60A5FA !important;
        }}

        button[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
        button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
        div[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"] {{
            color: #F8FAFC !important;
            font-size: 1.25rem !important;
        }}

        .block-container {{
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1380px !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border) !important;
        }}

        section[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{
            font-family: 'Inter', sans-serif !important;
        }}

        /* Sidebar Radio Navigation as Clean Pills */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
            background-color: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 10px !important;
            padding: 0.55rem 0.85rem !important;
            margin-bottom: 0.25rem !important;
            transition: all 0.18s ease !important;
            cursor: pointer !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            background-color: rgba(59, 130, 246, 0.12) !important;
            border-color: rgba(96, 165, 250, 0.25) !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has([aria-checked="true"]) {{
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.24), rgba(59, 130, 246, 0.08)) !important;
            border-color: var(--blue-primary) !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{
            color: var(--text-main) !important;
            font-weight: 500 !important;
            font-size: 0.92rem !important;
        }}

        /* Typography */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            color: var(--text-main) !important;
            font-family: 'Inter', sans-serif !important;
            letter-spacing: -0.02em;
        }}

        .stApp p, .stApp li, .stApp span, .stApp label {{
            color: var(--text-main);
        }}

        /* Equal-Height Cards & Consistent Vertical Gap Across Horizontal Rows */
        div[data-testid="stHorizontalBlock"] {{
            display: flex !important;
            align-items: stretch !important;
        }}

        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
        }}

        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.cva-card),
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stMetric"]),
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.cva-card) > div,
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stMetric"]) > div,
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"]:has(.cva-card),
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"]:has(div[data-testid="stMetric"]) {{
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            flex: 1 1 auto !important;
            width: 100% !important;
            row-gap: 0.85rem !important;
            gap: 0.85rem !important;
        }}

        /* Stretch every intermediate Streamlit wrapper around .cva-card or stMetric */
        div[data-testid="stHorizontalBlock"] div[data-testid="stElementContainer"]:has(.cva-card),
        div[data-testid="stHorizontalBlock"] div[data-testid="stElementContainer"]:has(.cva-card) div:has(.cva-card),
        div[data-testid="stHorizontalBlock"] div.element-container:has(.cva-card),
        div[data-testid="stHorizontalBlock"] div.element-container:has(.cva-card) div:has(.cva-card),
        div[data-testid="stHorizontalBlock"] div[data-testid="stElementContainer"]:has(div[data-testid="stMetric"]),
        div[data-testid="stHorizontalBlock"] div[data-testid="stElementContainer"]:has(div[data-testid="stMetric"]) div:has(div[data-testid="stMetric"]),
        div[data-testid="stHorizontalBlock"] div.element-container:has(div[data-testid="stMetric"]),
        div[data-testid="stHorizontalBlock"] div.element-container:has(div[data-testid="stMetric"]) div:has(div[data-testid="stMetric"]) {{
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            flex: 1 1 auto !important;
            width: 100% !important;
            margin-bottom: 0 !important;
        }}

        /* Stretch all custom cards inside columns to match the tallest card in the row */
        div[data-testid="stHorizontalBlock"] .cva-card {{
            flex: 1 1 auto !important;
            width: 100% !important;
            margin-bottom: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            box-sizing: border-box !important;
        }}

        /* Native Streamlit Metric Cards */
        div[data-testid="stMetric"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            padding: 1rem 1.15rem !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28) !important;
        }}

        div[data-testid="stHorizontalBlock"] div[data-testid="stMetric"] {{
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
        }}

        div[data-testid="stMetricLabel"] p {{
            color: var(--text-secondary) !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--text-main) !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
        }}

        /* Primary & Secondary Buttons (all Streamlit versions) */
        div.stButton > button,
        div.stDownloadButton > button,
        button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="baseButton-primary"],
        button[data-testid="baseButton-secondary"] {{
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important;
            padding: 0.55rem 1.15rem !important;
            transition: all 0.2s ease !important;
            opacity: 1 !important;
        }}

        /* Uniform Height & Full-Width Alignment for Side-by-Side Row Buttons */
        div[data-testid="stHorizontalBlock"] div[data-testid="stElementContainer"]:has(div.stButton),
        div[data-testid="stHorizontalBlock"] div[data-testid="stElementContainer"]:has(div.stDownloadButton) {{
            flex: 0 0 auto !important;
            width: 100% !important;
            margin-bottom: 0 !important;
        }}

        div[data-testid="stHorizontalBlock"] div.stButton > button,
        div[data-testid="stHorizontalBlock"] div.stDownloadButton > button {{
            min-height: 46px !important;
            height: 46px !important;
            width: 100% !important;
            white-space: nowrap !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}

        div.stButton > button[kind="primary"],
        div.stDownloadButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"],
        button[data-testid="baseButton-primary"] {{
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            border: 1px solid #60A5FA !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35) !important;
        }}

        div.stButton > button[kind="primary"] *,
        div.stDownloadButton > button[kind="primary"] *,
        button[data-testid="stBaseButton-primary"] *,
        button[data-testid="baseButton-primary"] * {{
            color: #FFFFFF !important;
        }}

        div.stButton > button[kind="primary"]:hover,
        div.stDownloadButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {{
            background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
            transform: translateY(-1px);
        }}

        div.stButton > button[kind="secondary"],
        div.stDownloadButton > button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="baseButton-secondary"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid rgba(96, 165, 250, 0.35) !important;
            color: var(--text-main) !important;
        }}

        div.stButton > button[kind="secondary"] *,
        div.stDownloadButton > button[kind="secondary"] *,
        button[data-testid="stBaseButton-secondary"] *,
        button[data-testid="baseButton-secondary"] * {{
            color: var(--text-main) !important;
        }}

        div.stButton > button[kind="secondary"]:hover,
        div.stDownloadButton > button[kind="secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover {{
            border-color: var(--blue-secondary) !important;
            background-color: var(--bg-card-hover) !important;
        }}

        /* File Uploader Dropzone & Browse Files Button */
        section[data-testid="stFileUploaderDropzone"] {{
            background-color: var(--bg-card) !important;
            border: 2px dashed rgba(96, 165, 250, 0.55) !important;
            border-radius: 16px !important;
            padding: 2.2rem 1.5rem !important;
            transition: border-color 0.2s ease, background-color 0.2s ease !important;
        }}

        section[data-testid="stFileUploaderDropzone"]:hover {{
            border-color: var(--blue-primary) !important;
            background-color: var(--bg-card-hover) !important;
        }}

        section[data-testid="stFileUploaderDropzone"] span,
        section[data-testid="stFileUploaderDropzone"] small {{
            color: #CBD5E1 !important;
        }}

        section[data-testid="stFileUploaderDropzone"] button,
        div[data-testid="stFileUploader"] button {{
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            border: 1px solid #60A5FA !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            padding: 0.5rem 1.2rem !important;
            font-weight: 600 !important;
            opacity: 1 !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35) !important;
        }}

        section[data-testid="stFileUploaderDropzone"] button *,
        div[data-testid="stFileUploader"] button * {{
            color: #FFFFFF !important;
        }}

        section[data-testid="stFileUploaderDropzone"] button:hover,
        div[data-testid="stFileUploader"] button:hover {{
            background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
        }}

        div[data-testid="stFileUploaderFile"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 0.5rem 0.85rem !important;
            color: var(--text-main) !important;
        }}

        div[data-testid="stFileUploaderFile"] svg {{
            fill: var(--text-main) !important;
        }}

        /* Text Input & Tabs Dark Styling */
        div[data-testid="stTextInput"] input {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text-main) !important;
            padding: 0.6rem 0.9rem !important;
        }}

        button[data-baseweb="tab"] {{
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--blue-secondary) !important;
            border-bottom-color: var(--blue-primary) !important;
        }}

        /* Selectbox & Dropdown Menu Dark Styling */
        div[data-baseweb="select"] > div {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text-main) !important;
        }}

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-main) !important;
        }}

        ul[role="listbox"] li,
        div[data-baseweb="menu"] li {{
            background-color: var(--bg-card) !important;
            color: var(--text-main) !important;
        }}

        ul[role="listbox"] li:hover,
        div[data-baseweb="menu"] li:hover,
        ul[role="listbox"] li[aria-selected="true"] {{
            background-color: var(--bg-card-hover) !important;
            color: var(--text-main) !important;
        }}

        /* Expanders & Containers */
        details[data-testid="stExpander"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }}

        /* Centered Responsive Video & Frame Display (Landscape + Portrait) */
        div[data-testid="stImage"],
        div[data-testid="stVideo"] {{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            background-color: #080D1A !important;
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
            overflow: hidden !important;
        }}

        div[data-testid="stImage"] > div,
        div[data-testid="stImageContainer"],
        div[data-testid="stVideo"] > div {{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
        }}

        div[data-testid="stImage"] img,
        div[data-testid="stVideo"] video {{
            max-height: 62vh !important;
            width: auto !important;
            max-width: 100% !important;
            height: auto !important;
            object-fit: contain !important;
            border-radius: 10px !important;
            margin: 0 auto !important;
        }}

        /* Custom Sports-Tech Cards */
        .cva-hero {{
            background: linear-gradient(135deg, #172338 0%, #0F172A 65%, rgba(59, 130, 246, 0.18) 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        }}

        .cva-badge {{
            display: inline-block;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }}

        .cva-badge-blue {{
            background: rgba(59, 130, 246, 0.18);
            color: #60A5FA;
            border: 1px solid rgba(96, 165, 250, 0.35);
        }}

        .cva-badge-green {{
            background: rgba(34, 197, 94, 0.16);
            color: #22C55E;
            border: 1px solid rgba(34, 197, 94, 0.35);
        }}

        .cva-badge-red {{
            background: rgba(239, 68, 68, 0.18);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}

        .cva-badge-orange {{
            background: rgba(245, 158, 11, 0.16);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }}

        .cva-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.24);
        }}

        .cva-card-title {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
        }}

        .cva-card-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-main);
            line-height: 1.2;
            word-break: normal;
        }}

        .cva-card-sub {{
            font-size: 0.82rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}

        .cva-invalid-banner {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, #172338 100%);
            border: 1.5px solid #EF4444;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.35rem;
        }}

        .cva-valid-banner {{
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.16) 0%, #172338 100%);
            border: 1px solid rgba(34, 197, 94, 0.55);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.35rem;
        }}

        .cva-section-header {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
            border-left: 4px solid var(--blue-primary);
            padding-left: 0.75rem;
            margin: 1.4rem 0 0.9rem 0;
        }}

        .cva-status-pill {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.65rem 0.9rem;
            margin-bottom: 0.55rem;
            font-size: 0.86rem;
        }}

        .cva-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin-bottom: 1rem;
        }}

        .cva-table th {{
            background: #111827;
            color: var(--text-secondary);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: left;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}

        .cva-table td {{
            padding: 0.72rem 1rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            color: var(--text-main);
            font-size: 0.9rem;
        }}

        /* Responsive Breakpoints (Tablet & Mobile) */
        @media (max-width: 1024px) {{
            .block-container {{
                padding-left: 1.1rem !important;
                padding-right: 1.1rem !important;
            }}
            .cva-hero {{
                padding: 1.35rem 1.4rem;
            }}
        }}

        @media (max-width: 768px) {{
            .block-container {{
                padding-top: 1rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
            }}
            div[data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
                gap: 0.65rem !important;
            }}
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}
            div[data-testid="stHorizontalBlock"] div.stButton > button,
            div[data-testid="stHorizontalBlock"] div.stDownloadButton > button {{
                height: auto !important;
                white-space: normal !important;
            }}
            div[data-testid="stHorizontalBlock"] .cva-card {{
                height: auto !important;
                min-height: auto !important;
                margin-bottom: 0.5rem !important;
            }}
            div[data-testid="stMetricValue"] {{
                font-size: 1.3rem !important;
            }}
            .cva-card-value {{
                font-size: 1.2rem;
            }}
            .cva-table th, .cva-table td {{
                padding: 0.55rem 0.65rem;
                font-size: 0.82rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title, subtitle=None):
    """Renders a styled left-accented section header with an optional subtitle."""
    sub_html = (
        f'<p style="margin: -0.55rem 0 0.95rem 0.95rem; color: #94A3B8; font-size: 0.9rem;">{html.escape(str(subtitle))}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div class="cva-section-header">{html.escape(str(title))}</div>
        {sub_html}
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title,
    value,
    subtitle="",
    accent_color=None,
    value_color=None,
    color=None,
    sub=None,
):
    """Renders a styled KPI / capability card matching the .cva-card CSS theme."""
    resolved_sub = sub if sub is not None else subtitle
    resolved_color = accent_color or value_color or color or PALETTE["text_main"]
    sub_html = (
        f'<div class="cva-card-sub">{html.escape(str(resolved_sub))}</div>'
        if resolved_sub
        else ""
    )

    st.markdown(
        f"""
        <div class="cva-card">
            <div>
                <div class="cva-card-title">{html.escape(str(title))}</div>
                <div class="cva-card-value" style="color: {resolved_color};">{html.escape(str(value))}</div>
            </div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_validity_banner(bundle):
    analysis_valid = bool(bundle.get("analysis_valid", False))
    impact_summary = bundle.get("impact_summary")

    if not analysis_valid:
        st.markdown(
            """
            <div class="cva-invalid-banner">
                <span class="cva-badge cva-badge-red">ANALYSIS INVALID</span>
                <h3 style="margin:0.15rem 0 0.45rem 0; color:#F8FAFC;">NO CONFIRMED BAT-BALL IMPACT</h3>
                <p style="margin:0; color:#CBD5E1; font-size:0.93rem;">
                    No confirmed bat-ball impact event was detected in this delivery.
                    All performance and technique scores are strictly marked <strong>N/A</strong>.
                    Detection, ball tracking, and posture telemetry remain available below for diagnostic inspection only.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        imp_frame = impact_summary.get("impact_frame", "N/A") if isinstance(impact_summary, dict) else "N/A"
        conf_frame = impact_summary.get("confirmation_frame", "N/A") if isinstance(impact_summary, dict) else "N/A"
        st.markdown(
            f"""
            <div class="cva-valid-banner">
                <span class="cva-badge cva-badge-green">ANALYSIS VALID</span>
                <h3 style="margin:0.15rem 0 0.35rem 0; color:#F8FAFC;">CONFIRMED BAT-BALL IMPACT</h3>
                <p style="margin:0; color:#CBD5E1; font-size:0.92rem;">
                    Impact isolated at source frame <strong>#{html.escape(str(imp_frame))}</strong>
                    (confirmed at frame <strong>#{html.escape(str(conf_frame))}</strong>).
                    Biomechanical joint measurements and technique evaluations are verified at impact.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_joint_label(joint_key):
    return str(joint_key).replace("_", " ").title()


# ============================================================
# PLOTLY VISUALIZATION BUILDERS (USING ONLY REAL ANALYZER DATA)
# ============================================================

def apply_plotly_dark_theme(fig, title_text, xaxis_title="", yaxis_title="", height=370):
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(family="Inter, sans-serif", size=15, color=PALETTE["text_main"]),
            x=0.01,
        ),
        paper_bgcolor=PALETTE["bg_card"],
        plot_bgcolor="#111A2E",
        font=dict(family="Inter, sans-serif", color=PALETTE["text_main"], size=12),
        height=height,
        margin=dict(l=24, r=24, t=52, b=36),
        xaxis=dict(
            title=xaxis_title,
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            tickfont=dict(color=PALETTE["text_secondary"]),
            title_font=dict(color=PALETTE["text_secondary"]),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            tickfont=dict(color=PALETTE["text_secondary"]),
            title_font=dict(color=PALETTE["text_secondary"]),
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.65)",
            bordercolor=PALETTE["border"],
            borderwidth=1,
            font=dict(color=PALETTE["text_main"]),
        ),
    )
    return fig


def build_joint_angle_chart(angles_dict):
    """CHART 1 — Joint Angle Comparison (degrees, valid joints only)."""
    if not PLOTLY_AVAILABLE or not isinstance(angles_dict, dict):
        return None

    ordered_joints = [
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]
    labels = []
    values = []
    for key in ordered_joints:
        raw_val = angles_dict.get(key)
        if is_finite_number(raw_val):
            labels.append(format_joint_label(key))
            values.append(round(float(raw_val), 1))

    # Also include any additional valid joint keys if returned by analyzer
    for key, raw_val in angles_dict.items():
        if key not in ordered_joints and is_finite_number(raw_val):
            labels.append(format_joint_label(key))
            values.append(round(float(raw_val), 1))

    if not labels:
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=[f"{v:.1f}°" for v in values],
                textposition="outside",
                marker=dict(
                    color=PALETTE["blue_primary"],
                    line=dict(color=PALETTE["blue_secondary"], width=1.2),
                ),
                hovertemplate="<b>%{x}</b><br>Angle: %{y:.1f}°<extra></extra>",
            )
        ]
    )
    apply_plotly_dark_theme(
        fig,
        "Joint Angle Comparison (Measured Joints)",
        xaxis_title="Joint",
        yaxis_title="Angle (Degrees °)",
        height=370,
    )
    fig.update_yaxes(range=[0, max(values) * 1.18])
    return fig


def build_technique_score_breakdown_chart(scores_dict, technique_score=None):
    """CHART 2 — Technique Score Breakdown (0–100 scale, valid scores only)."""
    if not PLOTLY_AVAILABLE or not isinstance(scores_dict, dict):
        return None

    labels = []
    values = []
    colors_list = []

    for joint, val in scores_dict.items():
        if is_finite_number(val):
            score_val = round(float(val), 1)
            labels.append(format_joint_label(joint))
            values.append(score_val)
            if score_val >= 80:
                colors_list.append(PALETTE["success"])
            elif score_val >= 50:
                colors_list.append(PALETTE["blue_primary"])
            else:
                colors_list.append(PALETTE["warning"])

    if is_finite_number(technique_score):
        t_val = round(float(technique_score), 1)
        labels.append("Technique Overall")
        values.append(t_val)
        colors_list.append(PALETTE["blue_secondary"])

    if not labels:
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=[f"{v:.1f}" for v in values],
                textposition="outside",
                marker=dict(color=colors_list),
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f} / 100<extra></extra>",
            )
        ]
    )
    apply_plotly_dark_theme(
        fig,
        "Technique Score Breakdown (0–100 Scale)",
        xaxis_title="Component",
        yaxis_title="Score (0–100)",
        height=370,
    )
    fig.update_yaxes(range=[0, 112])
    return fig


def build_technique_radar_chart(scores_dict):
    """CHART 3 — Technique Radar Chart (only existing comparable 0–100 dimensions)."""
    if not PLOTLY_AVAILABLE or not isinstance(scores_dict, dict):
        return None

    categories = []
    values = []
    for joint, val in scores_dict.items():
        if is_finite_number(val):
            categories.append(format_joint_label(joint))
            values.append(round(float(val), 1))

    # Radar requires at least 3 genuine comparable dimensions
    if len(categories) < 3:
        return None

    closed_categories = categories + [categories[0]]
    closed_values = values + [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=closed_values,
                theta=closed_categories,
                fill="toself",
                fillcolor="rgba(59, 130, 246, 0.25)",
                line=dict(color=PALETTE["blue_secondary"], width=2.5),
                marker=dict(color=PALETTE["blue_primary"], size=7),
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f} / 100<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text="Technique Balance Radar (Measured Joints)",
            font=dict(family="Inter, sans-serif", size=15, color=PALETTE["text_main"]),
            x=0.01,
        ),
        paper_bgcolor=PALETTE["bg_card"],
        font=dict(family="Inter, sans-serif", color=PALETTE["text_main"], size=12),
        height=390,
        margin=dict(l=45, r=45, t=55, b=40),
        polar=dict(
            bgcolor="#111A2E",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(148, 163, 184, 0.18)",
                tickfont=dict(color=PALETTE["text_secondary"], size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(148, 163, 184, 0.18)",
                tickfont=dict(color=PALETTE["text_main"], size=11),
            ),
        ),
        showlegend=False,
    )
    return fig


def build_phase_distribution_donut(phase_coverage):
    """CHART 5 — Proportional Phase Frame Distribution (only when real frame counts exist)."""
    if not PLOTLY_AVAILABLE or not isinstance(phase_coverage, dict):
        return None

    labels = []
    values = []
    for phase_name, count in phase_coverage.items():
        if is_finite_number(count) and float(count) > 0:
            labels.append(str(phase_name))
            values.append(int(round(float(count))))

    if not labels or sum(values) <= 0:
        return None

    donut_colors = [
        "#3B82F6",
        "#60A5FA",
        "#22C55E",
        "#F59E0B",
        "#818CF8",
        "#38BDF8",
        "#94A3B8",
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.56,
                marker=dict(
                    colors=donut_colors[: len(labels)],
                    line=dict(color=PALETTE["bg_card"], width=2),
                ),
                textinfo="label+percent",
                textfont=dict(color=PALETTE["text_main"], size=11),
                hovertemplate="<b>%{label}</b><br>Frames: %{value} (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text="Shot Phase Frame Distribution",
            font=dict(family="Inter, sans-serif", size=15, color=PALETTE["text_main"]),
            x=0.01,
        ),
        paper_bgcolor=PALETTE["bg_card"],
        font=dict(family="Inter, sans-serif", color=PALETTE["text_main"], size=12),
        height=370,
        margin=dict(l=20, r=20, t=52, b=24),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.65)",
            bordercolor=PALETTE["border"],
            borderwidth=1,
            font=dict(color=PALETTE["text_main"]),
        ),
    )
    return fig


def build_technique_weakness_comparison_chart(scores_dict):
    """CHART 6 — Sorted Technique Comparison to Identify Weaker Areas."""
    if not PLOTLY_AVAILABLE or not isinstance(scores_dict, dict):
        return None

    items = [
        (format_joint_label(k), round(float(v), 1))
        for k, v in scores_dict.items()
        if is_finite_number(v)
    ]
    if not items:
        return None

    # Sort ascending so weaker areas appear first for immediate coaching focus
    items.sort(key=lambda pair: pair[1])
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    bar_colors = [
        PALETTE["error"] if v < 50 else (PALETTE["warning"] if v < 80 else PALETTE["success"])
        for v in values
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                text=[f"{v:.1f} / 100" for v in values],
                textposition="outside",
                marker=dict(color=bar_colors),
                hovertemplate="<b>%{y}</b><br>Score: %{x:.1f} / 100<extra></extra>",
            )
        ]
    )
    apply_plotly_dark_theme(
        fig,
        "Technique Priority Comparison (Lowest to Highest Score)",
        xaxis_title="Joint Technique Score (0–100)",
        yaxis_title="Joint",
        height=max(300, 75 + len(labels) * 48),
    )
    fig.update_xaxes(range=[0, 115])
    return fig


# ============================================================
# VIDEO PROCESSING PIPELINE EXECUTION (UNCHANGED ANALYZER LOGIC)
# ============================================================

def render_live_frame_b64(placeholder, rgb_frame):
    """Render a resized RGB frame inline via Base64 to prevent Streamlit Cloud media 404s."""
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    b64_str = base64.b64encode(buffer.tobytes()).decode("ascii")
    placeholder.markdown(
        f'<div style="display:flex;justify-content:center;background:#070D19;border-radius:12px;padding:6px;border:1px solid rgba(96,165,250,0.18);">'
        f'<img src="data:image/jpeg;base64,{b64_str}" style="max-width:100%;max-height:62vh;border-radius:8px;object-fit:contain;" />'
        f'</div>',
        unsafe_allow_html=True,
    )


def execute_video_analysis_pipeline(video_meta, batting_stance):
    """
    Run CricketAnalyzer frame-by-frame with the exact original logic,
    live UI progress indicators, and export construction.
    """
    input_path = video_meta["path"]
    safe_name = video_meta["name"]
    temp_dir = os.path.dirname(input_path)
    human_shot_annotation = load_human_shot_annotation(safe_name)

    with st.spinner("Initializing AI Cricket Analyzer models..."):
        try:
            analyzer = CricketAnalyzer()
            analyzer.footwork.set_batting_stance(batting_stance)
        except Exception as exc:
            st.error(f"Failed to initialize analyzer: {exc}")
            return None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Unable to open uploaded video.")
        return None

    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0)
    if fps <= 0:
        fps = 30.0

    total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    duration = total_frames / fps

    # Metadata summary bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frame Rate", f"{fps:.2f} FPS")
    m2.metric("Source Frames", total_frames)
    m3.metric("Resolution", f"{width} × {height}")
    m4.metric("Duration", f"{duration:.2f}s")

    st.markdown(
        '<div id="live-analysis-anchor" style="scroll-margin-top: 1rem;"></div>',
        unsafe_allow_html=True,
    )

    progress_bar = st.progress(0)
    status_caption = st.empty()

    # Pipeline Stage Status Grid
    st.markdown("#### Pipeline Module Status")
    st_row1_c1, st_row1_c2, st_row1_c3 = st.columns(3)
    st_row2_c1, st_row2_c2, st_row2_c3 = st.columns(3)

    status_obj_det = st_row1_c1.empty()
    status_ball_trk = st_row1_c2.empty()
    status_bat_trk = st_row1_c3.empty()
    status_pose_est = st_row2_c1.empty()
    status_impact_det = st_row2_c2.empty()
    status_shot_ana = st_row2_c3.empty()

    # Live Video Feed & Live Telemetry
    feed_col, telem_col = st.columns([1.65, 1.0])
    with feed_col:
        video_placeholder = st.empty()
    with telem_col:
        live_phase = st.empty()
        live_shot = st.empty()
        live_score = st.empty()
        live_ball = st.empty()
        impact_placeholder = st.empty()

    joint_coverage_placeholder = st.empty()

    with st.expander("🔧 Technical & Detection Diagnostics", expanded=False):
        dbg1, dbg2 = st.columns(2)
        debug_frame = dbg1.empty()
        debug_ball = dbg2.empty()
        dbg3, dbg4 = st.columns(2)
        debug_bat = dbg3.empty()
        debug_batsman = dbg4.empty()

    # Frame/Impact/Shot State (exact mirror of original app.py)
    frame_number = 0
    analysis_frames = 0
    ball_detected_frames = 0
    best_score = 0.0
    best_report = None
    last_report = None
    last_completed_report = None
    impact_report = None
    impact_summary = None
    shot_not_played_report = None
    impact_events = []
    seen_impact_frames = set()
    last_annotated_rgb = None
    stop_reason = "Video ended"
    processing_error = None
    start_time = time.time()

    shot_type_tracker = (
        ShotTypeV2()
        if os.getenv("CRICKET_SHOT_TYPE_V2", "0") == "1"
        else None
    )
    impact_motion_tracker = (
        ImpactCenteredMotionV3()
        if os.getenv("CRICKET_IMPACT_MOTION_V3", "0") == "1"
        else None
    )
    bat_ball_v32_tracker = (
        BatBallMotionV32(frame_width=width, frame_height=height)
        if os.getenv("CRICKET_BAT_BALL_V32", "0") == "1"
        else None
    )
    detection_audit_v33_tracker = (
        DetectionEvidenceV33()
        if os.getenv("CRICKET_DETECTION_AUDIT_V33", "0") == "1"
        else None
    )

    completion_gate = ShotCompletionGate(
        grace_frames=getattr(analyzer.impact, "follow_through_grace_frames", 8),
        confirmation_frames=getattr(analyzer.impact, "confirmation_frames", 2),
        stable_frames=5,
    )

    def _status_pill_html(title, state_text, badge_class):
        return (
            f'<div class="cva-status-pill">'
            f'<span style="color:#CBD5E1;font-weight:600;">{html.escape(title)}</span>'
            f'<span class="cva-badge {badge_class}" style="margin:0;">{html.escape(state_text)}</span>'
            f'</div>'
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

            try:
                output, report = analyzer.process_frame(
                    frame, source_frame_number=frame_number
                )
            except Exception as exc:
                processing_error = f"Error processing frame {frame_number}: {exc}"
                stop_reason = "Analyzer error"
                break

            if not isinstance(report, dict):
                report = {}

            if shot_type_tracker is not None:
                shot_type_tracker.observe(
                    frame_number,
                    report,
                    analyzer.last_landmarks,
                )

            if impact_motion_tracker is not None:
                impact_motion_tracker.observe(
                    frame_number,
                    report,
                    analyzer.last_landmarks,
                )

            if bat_ball_v32_tracker is not None:
                fresh_bat_box = None
                if (
                    report.get("analysis_status") == "Analysis Working"
                    and getattr(analyzer, "frames_since_last_bat", -1) == 0
                    and getattr(analyzer, "last_valid_bat_box", None) is not None
                ):
                    fresh_bat_box = {
                        "fresh": True,
                        "box": list(analyzer.last_valid_bat_box),
                        "confidence": getattr(
                            analyzer, "last_valid_bat_confidence", 0.0
                        ),
                    }
                bat_ball_v32_tracker.observe(
                    frame_number,
                    report,
                    fresh_bat=fresh_bat_box,
                )

            if detection_audit_v33_tracker is not None:
                detection_audit_v33_tracker.observe(
                    frame_number,
                    report,
                    getattr(analyzer, "detection_audit_frame", None),
                )

            last_report = report

            if (
                str(report.get("phase", "")).strip().lower() == "shot not played"
                or str(report.get("shot", "")).strip().lower() == "shot not played"
            ):
                shot_not_played_report = deepcopy(report)

            current_score = safe_float(report.get("overall_score", 0))
            phase = str(report.get("phase", "Unknown"))
            shot = str(report.get("shot", "Unknown"))

            if current_score > best_score:
                best_score = current_score
                best_report = deepcopy(report)

            if is_completed_shot(phase, shot):
                last_completed_report = deepcopy(report)

            ball = report.get("ball")
            if isinstance(ball, dict):
                ball_detected_frames += 1

            if phase not in (
                "Waiting",
                "Unknown",
                "Crop Failed",
                "Pose Failed",
                "No Landmarks",
                "Angle Failed",
            ):
                analysis_frames += 1

            impact = report.get("impact", {})
            if not isinstance(impact, dict):
                impact = {}

            new_event = bool(impact.get("event", False))
            candidate_frame = (
                impact.get("impact_frame")
                or report.get("impact_frame")
                or frame_number
            )
            event_key = str(candidate_frame)

            is_fresh_event = new_event and event_key not in seen_impact_frames

            if is_fresh_event:
                seen_impact_frames.add(event_key)
                impact_summary = make_impact_summary(report, impact, frame_number)
                impact_report = deepcopy(report)
                impact_events.append(dict(impact_summary))

                coverage = impact_report.get("joint_measurement_reliability")
                if isinstance(coverage, dict):
                    measured = int(coverage.get("measured_joints", 0))
                    expected = int(coverage.get("expected_joints", 0))
                    percentage = safe_float(coverage.get("coverage_percent", 0))
                    missing = coverage.get("missing_joints", []) or []
                    details = coverage.get("missing_details", {}) or {}

                    with joint_coverage_placeholder.container():
                        render_section_header("Joint Measurement Coverage at Impact")
                        sc_c1, sc_c2 = st.columns(2)
                        sc_c1.metric(
                            "Impact Performance Score",
                            f"{impact_summary['score']:.1f}/100",
                        )
                        sc_c2.metric(
                            "Joint Coverage",
                            f"{measured}/{expected} ({percentage:.1f}%)",
                        )
                        if missing:
                            st.warning(
                                "Unavailable joint angles: "
                                + ", ".join(format_joint_label(n) for n in missing)
                            )
                            with st.expander("View Missing Landmark Details"):
                                for joint in missing:
                                    st.markdown(f"**{format_joint_label(joint)}**")
                                    joint_details = details.get(joint, {})
                                    for item in joint_details.get("failed_landmarks", []):
                                        l_name = item.get("name", "Unknown")
                                        reason = item.get("reason", "Unknown")
                                        vis = item.get("visibility")
                                        vis_text = (
                                            f"{float(vis):.3f}"
                                            if vis is not None
                                            else "Unavailable"
                                        )
                                        st.write(
                                            f"- {l_name}: {reason} (visibility: {vis_text})"
                                        )
                        else:
                            st.success("All configured joint angles were measured at impact.")

            impact_confirmed = (
                impact_summary is not None
                or getattr(analyzer, "impact_frame", None) is not None
            )

            stop_after_this_frame = completion_gate.update(
                frame_number, phase, shot, impact_confirmed=impact_confirmed
            )

            # Live Annotated Display Frame (fits portrait and landscape without cropping)
            if output is None:
                output = frame.copy()

            display_output = resize_frame_for_display(
                output, max_width=860, max_height=540
            )
            display_frame = cv2.cvtColor(display_output, cv2.COLOR_BGR2RGB)
            last_annotated_rgb = display_frame

            render_live_frame_b64(video_placeholder, display_frame)

            # Live Telemetry Cards
            live_phase.metric("Current Phase", phase)
            live_shot_status = (
                "Shot Played"
                if getattr(analyzer, "delivery_impact_confirmed", False)
                else "Shot Not Played"
            )
            live_shot.metric("Shot Status", live_shot_status)
            live_score.metric(
                "Live Frame Score",
                f"{current_score:.1f}/100"
                if getattr(analyzer, "delivery_impact_confirmed", False)
                else "N/A",
            )

            cricket_detector = getattr(analyzer, "cricket_detector", None)
            batsman_ok = bool(
                report.get("batsman_detected")
                or getattr(cricket_detector, "last_batsman", None) is not None
            )
            bat_ok = bool(getattr(cricket_detector, "last_bat", None) is not None)
            pose_ok = bool(
                report.get("pose_detected")
                or report.get("landmarks_detected")
                or getattr(analyzer, "last_landmarks", None) is not None
            )
            ball_ok = isinstance(ball, dict)

            status_obj_det.markdown(
                _status_pill_html(
                    "Object Detection",
                    "ACTIVE" if batsman_ok else "SEARCHING",
                    "cva-badge-green" if batsman_ok else "cva-badge-orange",
                ),
                unsafe_allow_html=True,
            )
            status_ball_trk.markdown(
                _status_pill_html(
                    "Ball Tracking",
                    "TRACKING" if ball_ok else "SCANNING",
                    "cva-badge-green" if ball_ok else "cva-badge-orange",
                ),
                unsafe_allow_html=True,
            )
            status_bat_trk.markdown(
                _status_pill_html(
                    "Bat Tracking",
                    "LOCKED" if bat_ok else "SEARCHING",
                    "cva-badge-green" if bat_ok else "cva-badge-orange",
                ),
                unsafe_allow_html=True,
            )
            status_pose_est.markdown(
                _status_pill_html(
                    "Pose Estimation",
                    "DETECTED" if pose_ok else "PENDING",
                    "cva-badge-green" if pose_ok else "cva-badge-orange",
                ),
                unsafe_allow_html=True,
            )
            status_impact_det.markdown(
                _status_pill_html(
                    "Impact Detection",
                    f"CONFIRMED #{impact_summary['impact_frame']}"
                    if impact_summary is not None
                    else "MONITORING",
                    "cva-badge-green" if impact_summary is not None else "cva-badge-blue",
                ),
                unsafe_allow_html=True,
            )
            status_shot_ana.markdown(
                _status_pill_html(
                    "Shot Analysis",
                    phase.upper(),
                    "cva-badge-blue",
                ),
                unsafe_allow_html=True,
            )

            if ball_ok:
                live_ball.success("⚾ Ball Detected")
                confidence = safe_float(ball.get("confidence", 0))
                debug_ball.write(
                    f"Ball confidence: {confidence:.3f} | Center: {ball.get('center')}"
                )
            else:
                live_ball.warning("⚾ Ball Not Detected")
                debug_ball.write("Ball: None")

            if impact_summary is not None:
                impact_placeholder.success(
                    f"💥 Confirmed impact at frame {impact_summary['impact_frame']} "
                    f"(confirmed at {impact_summary['confirmation_frame']})"
                )
            else:
                impact_placeholder.info("💥 Impact: Monitoring bat-ball trajectory")

            debug_frame.write(f"Frame: {frame_number} / {total_frames}")
            debug_bat.write("Bat memory: " + ("YES" if bat_ok else "NO"))
            debug_batsman.write("Batsman memory: " + ("YES" if batsman_ok else "NO"))

            if total_frames > 0:
                progress_bar.progress(min(frame_number / total_frames, 1.0))

            elapsed = max(time.time() - start_time, 0.001)
            status_caption.caption(
                f"Processing frame {frame_number}/{total_frames} | "
                f"Throughput: {frame_number / elapsed:.1f} FPS | "
                f"Elapsed: {elapsed:.1f}s"
            )

            if stop_after_this_frame:
                stop_reason = "Confirmed impact + stable completed shot"
                break

    finally:
        cap.release()

    if processing_error:
        st.error(processing_error)
        st.warning("Analysis stopped due to an error. Partial results are preserved.")

    if last_report is None:
        st.warning("No frames could be analyzed.")
        return None

    progress_bar.progress(1.0)
    elapsed_total = max(0.0, time.time() - start_time)
    detection_rate = 100.0 * ball_detected_frames / max(frame_number, 1)

    # Final Report Selection (exact mirror of original app.py)
    if impact_summary is not None:
        final_report = impact_report
    elif shot_not_played_report is not None:
        final_report = shot_not_played_report
    else:
        final_report = last_report

    analysis_valid = impact_summary is not None

    if analysis_valid:
        final_phase = impact_summary["phase"]
        final_shot = impact_summary["shot"]
        final_shot_status = "Shot Played"
        final_score = safe_float(impact_summary["score"])
    else:
        final_phase = final_report.get("phase", "Unknown")
        final_shot = final_report.get("shot", "Unknown")
        final_shot_status = "Shot Not Played"
        final_score = None

    score_evidence = build_score_evidence(final_report)

    swing_summary = analyzer.swing.get_summary()
    if isinstance(swing_summary, dict):
        confirmed_impact_frame = final_report.get("impact_frame")
        swing_summary["impact_alignment"] = analyzer.swing.get_impact_alignment(
            confirmed_impact_frame,
            window=8,
        )
        swing_summary["phase_coverage"] = analyzer.get_phase_coverage()

    shot_type_v2 = (
        shot_type_tracker.summarize() if shot_type_tracker is not None else None
    )
    impact_motion_v3 = (
        impact_motion_tracker.summarize() if impact_motion_tracker is not None else None
    )
    bat_ball_v32 = (
        bat_ball_v32_tracker.summarize() if bat_ball_v32_tracker is not None else None
    )
    detection_audit_v33 = (
        detection_audit_v33_tracker.summarize()
        if detection_audit_v33_tracker is not None
        else None
    )
    tracker_association_v34 = (
        TrackerAssociationV34(frame_width=width, frame_height=height).summarize(
            detection_audit_v33
        )
        if (
            os.getenv("CRICKET_TRACKER_ASSOC_V34", "0") == "1"
            and detection_audit_v33 is not None
        )
        else None
    )

    # Export Dictionary (100% identical structure to original app.py)
    export = {
        "video_name": safe_name,
        "batting_stance": batting_stance,
        "source_video": {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
        },
        "frames_processed": frame_number,
        "analysis_frames": analysis_frames,
        "ball_detected_frames": ball_detected_frames,
        "impact_count": len(impact_events),
        "impact_events": impact_events,
        "stop_reason": stop_reason,
        "final_phase": final_phase,
        "final_shot": final_shot,
        "shot_status": final_shot_status,
        "analysis_valid": analysis_valid,
        "validity_status": "VALID" if analysis_valid else "INVALID",
        "invalid_reason": (
            None if analysis_valid else "No confirmed bat-ball impact event"
        ),
        "final_score": final_score if analysis_valid else None,
        "final_report_scores_are_diagnostic_only": not analysis_valid,
        "swing_summary": swing_summary,
        "final_report": final_report,
        "human_reviewed_annotation": human_shot_annotation,
    }

    if score_evidence is not None:
        export["score_evidence"] = score_evidence
    if shot_type_v2 is not None:
        export["shot_type_v2"] = shot_type_v2
    if impact_motion_v3 is not None:
        export["impact_motion_v3"] = impact_motion_v3
    if bat_ball_v32 is not None:
        export["bat_ball_motion_v32"] = bat_ball_v32
    if detection_audit_v33 is not None:
        export["detection_evidence_v33"] = detection_audit_v33
    if tracker_association_v34 is not None:
        export["tracker_association_v34"] = tracker_association_v34

    # Generate JSON & PDF bytes using existing report generator
    json_str = json.dumps(export, indent=2, default=json_serializable)
    pdf_output_path = os.path.join(
        temp_dir, f"{os.path.splitext(safe_name)[0]}_report.pdf"
    )
    pdf_bytes = None
    pdf_error = None
    try:
        generate_pdf_report(export, pdf_output_path)
        with open(pdf_output_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
    except Exception as exc:
        pdf_error = str(exc)

    bundle = {
        "video_meta": video_meta,
        "batting_stance": batting_stance,
        "frames_processed": frame_number,
        "total_frames": total_frames,
        "analysis_frames": analysis_frames,
        "ball_detected_frames": ball_detected_frames,
        "detection_rate": detection_rate,
        "elapsed_seconds": elapsed_total,
        "stop_reason": stop_reason,
        "processing_error": processing_error,
        "analysis_valid": analysis_valid,
        "best_score": best_score,
        "final_score": final_score,
        "final_phase": final_phase,
        "final_shot": final_shot,
        "final_shot_status": final_shot_status,
        "impact_summary": impact_summary,
        "impact_events": impact_events,
        "final_report": final_report,
        "score_evidence": score_evidence,
        "swing_summary": swing_summary,
        "human_shot_annotation": human_shot_annotation,
        "shot_type_v2": shot_type_v2,
        "impact_motion_v3": impact_motion_v3,
        "bat_ball_v32": bat_ball_v32,
        "detection_audit_v33": detection_audit_v33,
        "tracker_association_v34": tracker_association_v34,
        "last_annotated_rgb": last_annotated_rgb,
        "export": export,
        "json_str": json_str,
        "pdf_bytes": pdf_bytes,
        "pdf_error": pdf_error,
    }
    return bundle


# ============================================================
# PAGE RENDERERS FOR ALL 8 SIDEBAR SECTIONS
# ============================================================

def render_dashboard_page():
    st.markdown(
        """
        <div class="cva-hero">
            <span class="cva-badge cva-badge-blue">AI SPORTS BIOMECHANICS SUITE</span>
            <h1 style="margin:0.1rem 0 0.5rem 0; font-size:2rem;">🏏 AI Cricket Performance Analyzer</h1>
            <p style="margin:0; color:#94A3B8; max-width:780px; font-size:1rem; line-height:1.55;">
                Computer-vision batting diagnostics powered by YOLO object detection, multi-stage ball tracking,
                MediaPipe skeletal pose estimation, bat-ball impact verification, and automated coaching reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_col1, cta_col2, _ = st.columns([1.2, 1.2, 2.6])
    with cta_col1:
        if st.button("📤 Upload Batting Video", type="primary", use_container_width=True):
            navigate_to("Upload Video")
            st.rerun()
    with cta_col2:
        if st.session_state.get("analysis_bundle") is not None:
            if st.button("🏆 View Latest Results", type="secondary", use_container_width=True):
                navigate_to("Performance Results")
                st.rerun()

    render_section_header("Platform Capabilities")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Object & Ball Tracking",
            "YOLO + Motion",
            "Detects batsman, bat, and high-speed deliveries with trajectory tracking.",
        )
    with c2:
        render_kpi_card(
            "Impact Verification",
            "Gated Validity",
            "Confirms genuine bat-ball contact before unlocking performance scores.",
        )
    with c3:
        render_kpi_card(
            "Pose Biomechanics",
            "8-Joint Kinematics",
            "Measures elbow, knee, shoulder, and hip angles against ideal ranges.",
        )
    with c4:
        render_kpi_card(
            "Coaching Exports",
            "PDF & JSON",
            "Generates structured diagnostic summaries and printable coaching reports.",
        )

    bundle = st.session_state.get("analysis_bundle")
    if bundle is not None:
        render_section_header("Active Session Overview")
        render_validity_banner(bundle)

        valid = bundle["analysis_valid"]
        score_txt = (
            f"{bundle['final_score']:.1f} / 100"
            if valid and bundle["final_score"] is not None
            else "N/A"
        )
        impact_txt = (
            f"CONFIRMED (Frame #{bundle['impact_summary']['impact_frame']})"
            if valid and bundle.get("impact_summary")
            else "NO CONFIRMED IMPACT"
        )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card(
                "Overall Performance",
                score_txt,
                bundle["video_meta"]["name"],
                accent_color=PALETTE["blue_secondary"] if valid else PALETTE["error"],
            )
        with k2:
            render_kpi_card(
                "Impact Status",
                impact_txt,
                f"Impact Events: {len(bundle['impact_events'])}",
                accent_color=PALETTE["success"] if valid else PALETTE["error"],
            )
        with k3:
            render_kpi_card(
                "Shot & Phase",
                bundle["final_shot_status"],
                f"Phase: {bundle['final_phase']}",
            )
        with k4:
            render_kpi_card(
                "Ball Detection Rate",
                f"{bundle['detection_rate']:.1f}%",
                f"{bundle['ball_detected_frames']}/{bundle['frames_processed']} frames",
            )

    recent = st.session_state.get("recent_analyses", [])
    if recent:
        render_section_header("Recent Session Analyses")
        rows_html = ""
        for item in reversed(recent):
            status_badge = (
                '<span class="cva-badge cva-badge-green" style="margin:0;">VALID</span>'
                if item["valid"]
                else '<span class="cva-badge cva-badge-red" style="margin:0;">INVALID</span>'
            )
            rows_html += (
                f"<tr>"
                f"<td><strong>{html.escape(item['video_name'])}</strong></td>"
                f"<td>{status_badge}</td>"
                f"<td>{html.escape(item['score_display'])}</td>"
                f"<td>{html.escape(item['shot_status'])}</td>"
                f"<td>{html.escape(item['phase'])}</td>"
                f"<td>{item['frames']}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <table class="cva-table">
                <thead>
                    <tr>
                        <th>Video File</th>
                        <th>Validity</th>
                        <th>Overall Score</th>
                        <th>Shot Status</th>
                        <th>Phase</th>
                        <th>Frames</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )


def render_upload_page():
    render_section_header(
        "Upload or Link Cricket Batting Video",
        "Upload a local clip (MP4, AVI, MOV) or paste a direct video / YouTube / Google Drive link. Original resolution is preserved.",
    )

    tab_file, tab_link = st.tabs(["📁 Upload Local File", "🔗 Paste Video Link"])

    with tab_file:
        uploaded_file = st.file_uploader(
            "Drag and drop a batting video file or browse from device",
            type=["mp4", "avi", "mov"],
            key="video_file_uploader",
        )

        if uploaded_file is not None:
            current_meta = st.session_state.get("uploaded_video_meta")
            # Persist uploaded buffer to temp file only when a new file is selected
            if (
                current_meta is None
                or current_meta.get("source_type") != "upload"
                or current_meta.get("name") != os.path.basename(uploaded_file.name)
                or current_meta.get("raw_size") != uploaded_file.size
            ):
                temp_dir = tempfile.mkdtemp(prefix="cricket_analysis_")
                safe_name = os.path.basename(uploaded_file.name)
                input_path = os.path.join(temp_dir, safe_name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                meta = inspect_video_metadata(input_path, safe_name, uploaded_file.size)
                if meta is not None:
                    meta["raw_size"] = uploaded_file.size
                    meta["source_type"] = "upload"
                    st.session_state["uploaded_video_meta"] = meta
                    st.session_state["analysis_bundle"] = None
                else:
                    st.error("Unable to read metadata from the uploaded video file.")

    with tab_link:
        url_col, btn_col = st.columns([3.5, 1.0])
        with url_col:
            video_url = st.text_input(
                "Video URL",
                placeholder="https://www.youtube.com/watch?v=... or direct .mp4 / Google Drive link",
                label_visibility="collapsed",
                key="video_url_input",
            )
        with btn_col:
            load_url_clicked = st.button("⬇ Load Video", type="primary", use_container_width=True)

        if load_url_clicked:
            if not video_url or not video_url.strip():
                st.warning("Please paste a valid video URL first.")
            else:
                with st.spinner("Fetching video from link..."):
                    try:
                        temp_dir = tempfile.mkdtemp(prefix="cricket_url_")
                        input_path, safe_name, file_size = download_video_from_url(video_url, temp_dir)
                        meta = inspect_video_metadata(input_path, safe_name, file_size)
                        if meta is not None:
                            meta["raw_size"] = file_size
                            meta["source_type"] = "url"
                            meta["source_url"] = video_url.strip()
                            st.session_state["uploaded_video_meta"] = meta
                            st.session_state["analysis_bundle"] = None
                            st.rerun()
                        else:
                            st.error("Downloaded file could not be opened as a valid video stream.")
                    except Exception as exc:
                        st.error(f"Failed to load video from URL: {exc}")

    meta = st.session_state.get("uploaded_video_meta")
    if meta is None:
        st.info("Upload a batting clip or paste a video link above to preview properties and start analysis.")
        return

    st.success(f"Loaded video: **{meta['name']}**")

    # Video Metadata Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Filename", meta["name"], f"{meta['size_mb']:.2f} MB")
    with c2:
        render_kpi_card("Duration", f"{meta['duration']:.2f} s", f"{meta['total_frames']} total frames")
    with c3:
        render_kpi_card("Resolution", f"{meta['width']} × {meta['height']}", "Original source scale")
    with c4:
        render_kpi_card("Frame Rate", f"{meta['fps']:.2f} FPS", "Detected stream rate")

    prev_col, ctrl_col = st.columns([1.45, 1.0])
    with prev_col:
        render_section_header("Uploaded Video Preview")
        st.video(meta["path"])

    with ctrl_col:
        render_section_header("Analysis Configuration")
        stance_options = ["Unknown", "Right-handed", "Left-handed"]
        current_stance = st.session_state.get("batting_stance", "Unknown")
        stance_idx = stance_options.index(current_stance) if current_stance in stance_options else 0

        selected_stance = st.selectbox(
            "Select Batting Stance",
            stance_options,
            index=stance_idx,
            help=(
                "Select the batsman's known batting stance. "
                "Used for informational front/back leg identification."
            ),
        )
        st.session_state["batting_stance"] = selected_stance

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

        if st.button("▶ Start Analysis", type="primary", use_container_width=True):
            st.session_state["trigger_analysis"] = True
            navigate_to("Video Analysis")
            st.rerun()

        if st.button("🗑 Remove / Replace Video", type="secondary", use_container_width=True):
            st.session_state["uploaded_video_meta"] = None
            st.session_state["analysis_bundle"] = None
            st.session_state["trigger_analysis"] = False
            st.rerun()


def render_video_analysis_page():
    render_section_header(
        "Live Video Analysis & Processing Telemetry",
        "Real-time object detection, ball tracking, pose estimation, and bat-ball impact verification.",
    )

    meta = st.session_state.get("uploaded_video_meta")
    if meta is None:
        st.warning("No video uploaded yet. Please upload a cricket video first.")
        if st.button("📤 Go to Upload Video", type="primary"):
            navigate_to("Upload Video")
            st.rerun()
        return

    # Execute pipeline if triggered
    if st.session_state.get("trigger_analysis", False):
        st.session_state["trigger_analysis"] = False
        bundle = execute_video_analysis_pipeline(
            meta, st.session_state.get("batting_stance", "Unknown")
        )
        if bundle is not None:
            st.session_state["analysis_bundle"] = bundle
            score_disp = (
                f"{bundle['final_score']:.1f}/100"
                if bundle["analysis_valid"] and bundle["final_score"] is not None
                else "N/A"
            )
            st.session_state["recent_analyses"].append(
                {
                    "video_name": meta["name"],
                    "valid": bundle["analysis_valid"],
                    "score_display": score_disp,
                    "shot_status": bundle["final_shot_status"],
                    "phase": str(bundle["final_phase"]),
                    "frames": bundle["frames_processed"],
                }
            )
            st.success("✅ Video analysis completed.")
            nav_c1, nav_c2, nav_c3 = st.columns(3)
            with nav_c1:
                if st.button("🏆 View Performance Results", type="primary", use_container_width=True):
                    navigate_to("Performance Results")
                    st.rerun()
            with nav_c2:
                if st.button("📐 Inspect Technique Analysis", type="secondary", use_container_width=True):
                    navigate_to("Technique Analysis")
                    st.rerun()
            with nav_c3:
                if st.button("📑 Open Reports & Downloads", type="secondary", use_container_width=True):
                    navigate_to("Reports and Downloads")
                    st.rerun()
        return

    # If already processed, display completed processing summary
    bundle = st.session_state.get("analysis_bundle")
    if bundle is None:
        st.info(f"Video **{meta['name']}** is ready for processing.")
        if st.button("▶ Start Analysis Now", type="primary"):
            st.session_state["trigger_analysis"] = True
            st.rerun()
        return

    render_validity_banner(bundle)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Frames Processed", f"{bundle['frames_processed']} / {bundle['total_frames']}")
    s2.metric("Active Analysis Frames", bundle["analysis_frames"])
    s3.metric("Ball Detection Rate", f"{bundle['detection_rate']:.1f}%")
    s4.metric("Processing Time", f"{bundle['elapsed_seconds']:.2f}s")

    st.caption(f"Completion condition: {bundle['stop_reason']}")

    if bundle.get("last_annotated_rgb") is not None:
        render_section_header("Final Annotated Frame Snapshot")
        st.image(
            bundle["last_annotated_rgb"],
            channels="RGB",
            use_container_width=False,
        )

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("🏆 Proceed to Performance Results", type="primary", use_container_width=True):
            navigate_to("Performance Results")
            st.rerun()
    with btn_c2:
        if st.button("🔄 Re-run Analysis", type="secondary", use_container_width=True):
            st.session_state["trigger_analysis"] = True
            st.rerun()


def render_performance_results_page():
    render_section_header(
        "Cricket Performance Results",
        "Verified performance metrics gated by bat-ball impact confirmation.",
    )

    bundle = st.session_state.get("analysis_bundle")
    if bundle is None:
        st.info("No completed analysis available. Upload and analyze a video to view performance results.")
        return

    render_validity_banner(bundle)

    valid = bundle["analysis_valid"]
    final_report = bundle.get("final_report", {}) or {}
    footwork = final_report.get("footwork", {})
    impact_summary = bundle.get("impact_summary")

    overall_display = (
        f"{bundle['final_score']:.1f} / 100"
        if valid and bundle["final_score"] is not None
        else "N/A"
    )
    impact_display = (
        f"YES — Frame #{impact_summary['impact_frame']}"
        if valid and isinstance(impact_summary, dict)
        else "NO CONFIRMED BAT-BALL IMPACT"
    )

    raw_tech_score = final_report.get("technique_score")
    tech_display = (
        f"{float(raw_tech_score):.1f} / 100"
        if valid and is_finite_number(raw_tech_score)
        else "N/A"
    )

    raw_fw_score = (
        footwork.get("footwork_score") if isinstance(footwork, dict) else None
    )
    fw_display = (
        f"{raw_fw_score} / 100"
        if valid and is_finite_number(raw_fw_score)
        else "N/A"
    )

    human_anno = bundle.get("human_shot_annotation", {})
    shot_label_display = (
        human_anno["shot_type"]
        if isinstance(human_anno, dict) and human_anno.get("status") == "reviewed"
        else str(bundle["final_shot"])
    )

    # Top KPI Row
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        render_kpi_card(
            "Overall Performance Score",
            overall_display,
            "Verified at impact frame" if valid else "Invalid — No confirmed impact",
            accent_color=PALETTE["blue_secondary"] if valid else PALETTE["error"],
        )
    with r1_c2:
        render_kpi_card(
            "Impact Status",
            impact_display,
            f"Impact Events Recorded: {len(bundle['impact_events'])}",
            accent_color=PALETTE["success"] if valid else PALETTE["error"],
        )
    with r1_c3:
        render_kpi_card(
            "Shot Classification",
            shot_label_display,
            f"Status: {bundle['final_shot_status']} | Phase: {bundle['final_phase']}",
        )

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        render_kpi_card(
            "Technique Score",
            tech_display,
            "Biomechanical joint evaluation" if valid else "Requires confirmed impact",
            accent_color=PALETTE["text_main"] if valid else PALETTE["error"],
        )
    with r2_c2:
        render_kpi_card(
            "Footwork Stability",
            fw_display,
            "Stride & balance rating" if valid else "Requires confirmed impact",
            accent_color=PALETTE["text_main"] if valid else PALETTE["error"],
        )
    with r2_c3:
        render_kpi_card(
            "Diagnostic Ball Detection",
            f"{bundle['detection_rate']:.1f}%",
            f"{bundle['ball_detected_frames']} of {bundle['frames_processed']} frames (diagnostic)",
        )

    # Score Evidence & Reliability Section
    score_evidence = bundle.get("score_evidence")
    if isinstance(score_evidence, dict):
        render_section_header("Score Input Coverage & Evidence")
        ev1, ev2 = st.columns(2)
        ev1.metric(
            "Joint Scores Used",
            f"{score_evidence.get('scored_joints', 0)}/{score_evidence.get('expected_joints', 0)}",
        )
        ev2.metric(
            "Scoring Input Coverage",
            f"{safe_float(score_evidence.get('score_input_coverage_percent', 0)):.1f}%",
        )
        for warning in score_evidence.get("warnings", []):
            st.warning(warning)

        missing_scores = score_evidence.get("missing_joint_scores", [])
        if missing_scores:
            st.caption(
                "Unscored / occluded joints (excluded from average): "
                + ", ".join(format_joint_label(j) for j in missing_scores)
            )

    # Charts Section (Strictly Gated on Impact Validity)
    if not valid:
        st.info(
            "Score-dependent performance charts are hidden because no bat-ball impact was confirmed. "
            "Diagnostic joint angles and telemetry can be inspected in Technique Analysis and Shot Analysis."
        )
        return

    render_section_header("Performance Visualizations")
    scores_dict = final_report.get("scores", {})
    angles_dict = final_report.get("angles", {})

    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        fig_scores = build_technique_score_breakdown_chart(
            scores_dict, final_report.get("technique_score")
        )
        if fig_scores is not None:
            st.plotly_chart(fig_scores, use_container_width=True)
    with ch_col2:
        fig_radar = build_technique_radar_chart(scores_dict)
        if fig_radar is not None:
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            fig_angles = build_joint_angle_chart(angles_dict)
            if fig_angles is not None:
                st.plotly_chart(fig_angles, use_container_width=True)


def render_technique_analysis_page():
    render_section_header(
        "Biomechanical Technique Analysis",
        "Joint angle measurements, joint scores, detected posture flaws, and coaching feedback.",
    )

    bundle = st.session_state.get("analysis_bundle")
    if bundle is None:
        st.info("No analysis data available yet. Please analyze a video first.")
        return

    render_validity_banner(bundle)

    valid = bundle["analysis_valid"]
    final_report = bundle.get("final_report", {}) or {}
    angles = final_report.get("angles", {}) if isinstance(final_report.get("angles"), dict) else {}
    scores = final_report.get("scores", {}) if isinstance(final_report.get("scores"), dict) else {}
    mistakes = final_report.get("mistakes", []) or []

    # Joint Angles & Scores Table
    render_section_header(
        "Joint Angles & Technique Scores"
        + ("" if valid else " (Diagnostic Angles Only — Scores N/A)")
    )

    expected_joints = [
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]
    all_keys = list(dict.fromkeys(expected_joints + list(angles.keys()) + list(scores.keys())))

    table_rows = ""
    for joint in all_keys:
        ang_val = angles.get(joint)
        ang_str = f"{float(ang_val):.1f}°" if is_finite_number(ang_val) else "Not Measured"

        if valid and is_finite_number(scores.get(joint)):
            sc_num = float(scores[joint])
            sc_str = f"{sc_num:.1f} / 100"
            if sc_num >= 80:
                status_html = '<span class="cva-badge cva-badge-green" style="margin:0;">Optimal</span>'
            elif sc_num >= 50:
                status_html = '<span class="cva-badge cva-badge-orange" style="margin:0;">Acceptable</span>'
            else:
                status_html = '<span class="cva-badge cva-badge-red" style="margin:0;">Needs Work</span>'
        else:
            sc_str = "N/A"
            status_html = (
                '<span class="cva-badge cva-badge-blue" style="margin:0;">Diagnostic Only</span>'
                if (not valid and is_finite_number(ang_val))
                else '<span style="color:#94A3B8;">Unavailable</span>'
            )

        table_rows += (
            f"<tr>"
            f"<td><strong>{html.escape(format_joint_label(joint))}</strong></td>"
            f"<td>{html.escape(ang_str)}</td>"
            f"<td>{html.escape(sc_str)}</td>"
            f"<td>{status_html}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <table class="cva-table">
            <thead>
                <tr>
                    <th>Joint</th>
                    <th>Measured Angle (°)</th>
                    <th>Technique Score</th>
                    <th>Evaluation</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # Coaching Feedback & Technical Mistakes
    render_section_header("Detected Technical Mistakes & Coaching Feedback")
    if mistakes:
        for item in mistakes:
            text = str(item)
            if "excellent" in text.lower():
                st.success(f"✅ {text}")
            else:
                st.warning(f"⚠️ {text}")
    else:
        st.success("✅ No major technique issues detected in the evaluated frame.")

    # Charts 1, 2, 3, 6 (and explicit limitation note for Chart 4)
    render_section_header("Biomechanical Visualizations")

    fig_angles = build_joint_angle_chart(angles)
    if fig_angles is not None:
        st.plotly_chart(fig_angles, use_container_width=True)
    else:
        st.info("No valid joint angle measurements are available to plot.")

    if valid:
        col_left, col_right = st.columns(2)
        with col_left:
            fig_breakdown = build_technique_score_breakdown_chart(
                scores, final_report.get("technique_score")
            )
            if fig_breakdown is not None:
                st.plotly_chart(fig_breakdown, use_container_width=True)
        with col_right:
            fig_radar = build_technique_radar_chart(scores)
            if fig_radar is not None:
                st.plotly_chart(fig_radar, use_container_width=True)

        fig_weakness = build_technique_weakness_comparison_chart(scores)
        if fig_weakness is not None:
            st.plotly_chart(fig_weakness, use_container_width=True)
    else:
        st.info(
            "Technique score breakdown, radar, and priority comparison charts are hidden "
            "because no confirmed bat-ball impact was detected."
        )

    # Chart 4 Limitation Note (do not fabricate frame-level joint history)
    st.caption(
        "ℹ️ Joint Movement Over Time chart omitted: the current analyzer export records joint angles "
        "at the confirmed impact/evaluated snapshot rather than persisting a frame-by-frame angle time-series."
    )


def render_footwork_analysis_page():
    render_section_header(
        "Footwork & Balance Analysis",
        "Lower-body stride stability, hip alignment, and coaching observations.",
    )

    bundle = st.session_state.get("analysis_bundle")
    if bundle is None:
        st.info("No analysis data available yet. Please analyze a video first.")
        return

    render_validity_banner(bundle)

    valid = bundle["analysis_valid"]
    final_report = bundle.get("final_report", {}) or {}
    footwork = final_report.get("footwork", {})

    if isinstance(footwork, dict):
        fw_score = footwork.get("footwork_score")
        fw_display = (
            f"{fw_score} / 100"
            if valid and is_finite_number(fw_score)
            else "N/A"
        )

        c1, c2 = st.columns(2)
        with c1:
            render_kpi_card(
                "Footwork Stability Score",
                fw_display,
                "Evaluated at impact" if valid else "Invalid — No confirmed bat-ball impact",
                accent_color=PALETTE["success"] if valid else PALETTE["error"],
            )
        with c2:
            stance_used = bundle.get("batting_stance", "Unknown")
            render_kpi_card(
                "Configured Batting Stance",
                stance_used,
                "Front/back leg reference setting",
            )

        render_section_header("Footwork Observations & Feedback")
        feedback_list = footwork.get("feedback", []) or []
        if feedback_list:
            for fb in feedback_list:
                fb_str = str(fb)
                if fb_str.startswith("✓") or "good" in fb_str.lower() or "stable" in fb_str.lower():
                    st.success(fb_str)
                elif fb_str.startswith("✗"):
                    st.warning(fb_str)
                else:
                    st.info(fb_str)
        else:
            st.write("No specific footwork feedback messages recorded.")

        # Existing non-score diagnostic footwork measurements
        def _format_footwork_metric(val) -> str:
            if isinstance(val, float):
                return f"{val:.3f}"
            if isinstance(val, dict):
                # Handle stance summary dict if present
                if "batting_stance" in val and "status" not in val:
                    stance_label = val.get("batting_stance", "Unknown")
                    geom = val.get("geometry_status", "")
                    return f"{stance_label} ({geom})" if geom else str(stance_label)

                status = str(val.get("status", "Measured"))
                details = []

                metric_labels = [
                    ("relative_motion_per_frame", "Motion", "{:.4f}/frame"),
                    ("forward_step_ratio", "Forward Step Ratio", "{:.4f}"),
                    ("relative_shift_per_frame", "Shift", "{:.4f}/frame"),
                    ("horizontal_shift_per_frame", "Horizontal Shift", "{:.4f}/frame"),
                ]
                for field, label, fmt in metric_labels:
                    if val.get(field) is not None:
                        details.append(f"{label}: {fmt.format(float(val[field]))}")

                if val.get("samples"):
                    details.append(f"Samples: {val['samples']}")

                diag = val.get("diagnostics")
                if isinstance(diag, dict) and diag.get("separation_ratio") is not None:
                    details.append(f"Ankle Separation Ratio: {float(diag['separation_ratio']):.2f}")

                return f"{status} ({' | '.join(details)})" if details else status
            if isinstance(val, list):
                return ", ".join(str(item) for item in val)
            return str(val)

        filtered_metrics = {
            k: v
            for k, v in footwork.items()
            if k != "feedback" and (valid or "score" not in str(k).lower())
        }
        if filtered_metrics:
            render_section_header("Footwork Kinematic Measurements")
            rows_html = ""
            for key, val in filtered_metrics.items():
                val_str = _format_footwork_metric(val)
                rows_html += (
                    f"<tr>"
                    f"<td><strong>{html.escape(format_joint_label(key))}</strong></td>"
                    f"<td>{html.escape(val_str)}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"""
                <table class="cva-table">
                    <thead>
                        <tr>
                            <th>Footwork Parameter</th>
                            <th>Measured Value</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

    elif isinstance(footwork, list) and footwork:
        for item in footwork:
            st.write(f"• {item}")
    else:
        st.info("No footwork information available for this session.")


def render_shot_analysis_page():
    render_section_header(
        "Shot, Phase & Impact Analysis",
        "Stroke classification, temporal phase transitions, bat-ball impact confirmation, and swing kinematics.",
    )

    bundle = st.session_state.get("analysis_bundle")
    if bundle is None:
        st.info("No analysis data available yet. Please analyze a video first.")
        return

    render_validity_banner(bundle)

    valid = bundle["analysis_valid"]
    final_report = bundle.get("final_report", {}) or {}
    impact_summary = bundle.get("impact_summary")
    human_anno = bundle.get("human_shot_annotation", {})

    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi_card("Shot Status", bundle["final_shot_status"])
    with c2:
        render_kpi_card("Evaluated Phase", str(bundle["final_phase"]))
    with c3:
        shot_type_val = (
            human_anno["shot_type"]
            if isinstance(human_anno, dict) and human_anno.get("status") == "reviewed"
            else str(bundle["final_shot"])
        )
        render_kpi_card("Shot Type", shot_type_val)

    # Impact Confirmation Section
    render_section_header("Bat-Ball Impact Verification")
    if valid and isinstance(impact_summary, dict):
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Estimated Impact Frame", f"#{impact_summary.get('impact_frame', 'N/A')}")
        ic2.metric("Confirmation Frame", f"#{impact_summary.get('confirmation_frame', 'N/A')}")
        ic3.metric("Impact Phase", str(impact_summary.get("phase", "Impact")))
        ic4.metric("Evidence Status", str(impact_summary.get("evidence_status", "Confirmed")))
    else:
        st.warning(
            "No confirmed bat-ball impact event was detected in this video. "
            "The full video was processed without early impact-completion cutoff."
        )

    # Swing & Wrist Movement Section
    render_section_header("Swing & Wrist Kinematics")
    swing = final_report.get("swing", {})
    swing_summary = bundle.get("swing_summary", {})

    if isinstance(swing, dict) and swing:
        sw_cols = st.columns(min(len(swing), 4))
        for idx, (k, v) in enumerate(swing.items()):
            with sw_cols[idx % len(sw_cols)]:
                disp_v = f"{float(v):.2f}" if isinstance(v, float) else str(v)
                render_kpi_card(format_joint_label(k), disp_v)
    else:
        st.info("No impact-frame wrist swing metrics recorded.")

    if isinstance(swing_summary, dict) and swing_summary:
        phase_cov = swing_summary.get("phase_coverage")
        fig_donut = build_phase_distribution_donut(phase_cov)
        if fig_donut is not None:
            st.plotly_chart(fig_donut, use_container_width=True)

        with st.expander("View Full Swing & Alignment Summary"):
            st.json(swing_summary)

    # Optional Experimental Modules (only displayed if enabled in environment)
    exp_modules = [
        ("Shot Classification v2", bundle.get("shot_type_v2")),
        ("Impact-Centered Motion v3", bundle.get("impact_motion_v3")),
        ("Bat & Ball Motion v3.2", bundle.get("bat_ball_v32")),
        ("Detection Evidence v3.3", bundle.get("detection_audit_v33")),
        ("Tracker Association v3.4", bundle.get("tracker_association_v34")),
    ]
    active_exp = [(name, data) for name, data in exp_modules if data is not None]
    if active_exp:
        render_section_header("Experimental Research Telemetry")
        for mod_name, mod_data in active_exp:
            with st.expander(f"🧪 {mod_name}"):
                st.json(mod_data)


def render_reports_downloads_page():
    render_section_header(
        "Reports & Exports",
        "Download the machine-readable JSON summary and formatted PDF performance report.",
    )

    bundle = st.session_state.get("analysis_bundle")
    if bundle is None:
        st.info("No analysis report generated yet. Please upload and analyze a video first.")
        return

    render_validity_banner(bundle)

    safe_name = bundle["video_meta"]["name"]
    base_stem = os.path.splitext(safe_name)[0]
    export = bundle["export"]

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        render_kpi_card(
            "PDF Performance Report",
            f"{base_stem}_report.pdf",
            "Formatted coaching & biomechanical summary via ReportLab.",
        )
        if bundle.get("pdf_bytes") is not None:
            st.download_button(
                "📄 Download PDF Performance Report",
                data=bundle["pdf_bytes"],
                file_name=f"{base_stem}_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        else:
            st.error(f"PDF generation unavailable: {bundle.get('pdf_error', 'Unknown error')}")

    with dl_col2:
        render_kpi_card(
            "Analysis Summary JSON",
            f"{base_stem}_summary.json",
            "Complete session export with validity flags and diagnostic payloads.",
        )
        st.download_button(
            "⬇ Download Analysis JSON",
            data=bundle["json_str"],
            file_name=f"{base_stem}_summary.json",
            mime="application/json",
            type="secondary",
            use_container_width=True,
        )

    # Structured Inline Report Preview
    render_section_header("Report Executive Preview")
    valid = bundle["analysis_valid"]
    score_preview = (
        f"{bundle['final_score']:.1f} / 100"
        if valid and bundle["final_score"] is not None
        else "N/A (Analysis Invalid — No Confirmed Impact)"
    )

    # Pull the human-annotated shot type (from annotation dict / CSV lookup)
    human_ann = (
        export.get("human_reviewed_annotation")
        or bundle.get("human_reviewed_annotation")
        or {}
    )
    human_shot_type = (
        human_ann.get("shot_type")
        or human_ann.get("shot_label")
        or export.get("human_shot_type")
        or export.get("annotated_shot_type")
        or bundle.get("human_shot_type")
        or bundle.get("annotated_shot_type")
        or "Not Annotated"
    )

    # Keep exported JSON & PDF report in sync with the preview table
    export["shot_type"] = str(human_shot_type)

    preview_rows = [
        ("Video Source", safe_name),
        ("Validity Status", export.get("validity_status", "INVALID")),
        ("Overall Performance Score", score_preview),
        ("Shot Status", bundle["final_shot_status"]),
        ("Shot Type", str(human_shot_type)),
        ("Evaluated Phase", str(bundle["final_phase"])),
        ("Frames Processed", f"{bundle['frames_processed']} / {bundle['total_frames']}"),
        ("Ball Detection Rate", f"{bundle['detection_rate']:.1f}%"),
        ("Impact Events", str(len(bundle["impact_events"]))),
    ]
    rows_html = "".join(
        f"<tr><td><strong>{html.escape(k)}</strong></td><td>{html.escape(v)}</td></tr>"
        for k, v in preview_rows
    )
    st.markdown(
        f"""
        <table class="cva-table">
            <thead>
                <tr>
                    <th>Report Field</th>
                    <th>Exported Value</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Inspect Raw Exported JSON Payload", expanded=False):
        st.code(bundle["json_str"], language="json")


# ============================================================
# APPLICATION ENTRYPOINT
# ============================================================

st.set_page_config(
    page_title="AI Cricket Performance Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
inject_custom_css()

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.4rem 0.2rem 1rem 0.2rem; border-bottom: 1px solid rgba(96, 165, 250, 0.18); margin-bottom: 1rem;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: #60A5FA; text-transform: uppercase;">
                SPORTS ANALYTICS ENGINE
            </div>
            <div style="font-size: 1.22rem; font-weight: 800; color: #F8FAFC; margin-top: 0.15rem;">
                🏏 CricketVision AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_labels = [f"{icon}  {name}" for name, icon in NAV_SECTIONS]
    nav_names = [name for name, _ in NAV_SECTIONS]

    current_nav = st.session_state.get("active_nav", "Dashboard")
    current_idx = nav_names.index(current_nav) if current_nav in nav_names else 0

    selected_label = st.radio(
        "Navigation",
        nav_labels,
        index=current_idx,
        label_visibility="collapsed",
    )
    selected_name = nav_names[nav_labels.index(selected_label)]
    if selected_name != st.session_state["active_nav"]:
        st.session_state["active_nav"] = selected_name
        st.rerun()

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

    # Sidebar Session Status Indicator
    active_meta = st.session_state.get("uploaded_video_meta")
    active_bundle = st.session_state.get("analysis_bundle")
    if active_bundle is not None:
        badge_cls = "cva-badge-green" if active_bundle["analysis_valid"] else "cva-badge-red"
        badge_txt = "VALID ANALYSIS" if active_bundle["analysis_valid"] else "INVALID (NO IMPACT)"
        st.markdown(
            f"""
            <div class="cva-card" style="padding:0.9rem 1rem;">
                <div class="cva-card-title">Active Session</div>
                <div style="font-size:0.88rem; font-weight:600; color:#F8FAFC; word-break:break-word;">
                    {html.escape(active_meta['name'] if active_meta else 'Analyzed Clip')}
                </div>
                <div style="margin-top:0.5rem;">
                    <span class="cva-badge {badge_cls}" style="margin:0;">{badge_txt}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif active_meta is not None:
        st.markdown(
            f"""
            <div class="cva-card" style="padding:0.9rem 1rem;">
                <div class="cva-card-title">Video Staged</div>
                <div style="font-size:0.88rem; font-weight:600; color:#F8FAFC; word-break:break-word;">
                    {html.escape(active_meta['name'])}
                </div>
                <div class="cva-card-sub">Ready to start analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Route to Active View
active_section = st.session_state.get("active_nav", "Dashboard")

if active_section == "Dashboard":
    render_dashboard_page()
elif active_section == "Upload Video":
    render_upload_page()
elif active_section == "Video Analysis":
    render_video_analysis_page()
elif active_section == "Performance Results":
    render_performance_results_page()
elif active_section == "Technique Analysis":
    render_technique_analysis_page()
elif active_section == "Footwork Analysis":
    render_footwork_analysis_page()
elif active_section == "Shot Analysis":
    render_shot_analysis_page()
elif active_section == "Reports and Downloads":
    render_reports_downloads_page()