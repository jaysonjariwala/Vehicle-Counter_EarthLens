# ============================================================
#  EarthLens - Vehicle Counter
#  Team: EarthLens
#  Members:
#    Jayson JJ       - 2025813005
#    Gangothri CJ     - 2025813004
#    Padmaja Y       - 2025813002
#  Course: SMAI (CS7.403) Assignment 3 - T5.4
#  Institute: IIIT Hyderabad
#  Academic Year: 2025-26
# ============================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import requests
from io import BytesIO
from ultralytics import YOLO
import plotly.graph_objects as go

st.set_page_config(
    page_title="EarthLens - Vehicle Counter",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8e8e8; }
    .main-title {
        font-size: 2.4rem; font-weight: 700; color: #f5c518;
        text-align: center; margin-bottom: 4px;
    }
    .sub-title {
        font-size: 1rem; font-weight: 700; color: #ffffff;
        text-align: center; margin-bottom: 2px;
    }
    .prof-title {
        font-size: 1rem; font-weight: 400; color: #ffffff;
        text-align: center; margin-bottom: 6px;
    }
    .team-line {
        font-size: 1rem; font-weight: 400; color: #ffffff;
        text-align: center; margin-bottom: 4px;
    }
    .team-name {
        font-size: 1.05rem; font-weight: 700; color: #ffffff;
        text-align: center; margin-bottom: 18px;
    }
    .count-box {
        background: #1a1d27; border: 2px solid #f5c518;
        border-radius: 10px; padding: 18px; text-align: center; margin: 8px 0;
    }
    .count-box-green {
        background: #1a1d27; border: 2px solid #00c875;
        border-radius: 10px; padding: 18px; text-align: center; margin: 8px 0;
    }
    .count-label {
        font-size: 0.85rem; color: #aaaaaa;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .count-num { font-size: 2.6rem; font-weight: 800; color: #f5c518; }
    .count-num-green { font-size: 2.6rem; font-weight: 800; color: #00c875; }
    .cat-box {
        background: #1a1d27; border-left: 4px solid #4fc3f7;
        border-radius: 6px; padding: 12px 16px; margin: 6px 0;
        display: flex; justify-content: space-between;
    }
    .cat-box-green {
        background: #1a1d27; border-left: 4px solid #00c875;
        border-radius: 6px; padding: 12px 16px; margin: 6px 0;
        display: flex; justify-content: space-between;
    }
    .cat-name { color: #e8e8e8; font-size: 0.95rem; }
    .cat-val  { color: #4fc3f7; font-weight: 700; font-size: 1.1rem; }
    .cat-val-green { color: #00c875; font-weight: 700; font-size: 1.1rem; }
    .section-hdr {
        font-size: 1.1rem; font-weight: 600; color: #f5c518;
        border-bottom: 1px solid #333; padding-bottom: 6px; margin: 20px 0 10px 0;
    }
    .section-hdr-green {
        font-size: 1.1rem; font-weight: 600; color: #00c875;
        border-bottom: 1px solid #333; padding-bottom: 6px; margin: 20px 0 10px 0;
    }
    .tip-box {
        background: #1a1d27; border-left: 3px solid #f5c518;
        border-radius: 4px; padding: 8px 12px; font-size: 0.82rem;
        color: #cccccc; margin-top: 4px; margin-bottom: 10px;
    }
    .info-box {
        background: #1a1d27; border: 1px solid #333;
        border-radius: 6px; padding: 10px 14px;
        font-size: 0.85rem; color: #cccccc; margin: 8px 0;
    }
    .track-note {
        background: #0d1f17; border: 1px solid #00c875;
        border-radius: 6px; padding: 8px 12px;
        font-size: 0.8rem; color: #aaaaaa; margin: 6px 0;
    }
    div[data-testid="stButton"] > button {
        background: #f5c518; color: #0f1117; font-weight: 700;
        border: none; border-radius: 6px; width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        background: #ffd740; color: #0f1117;
    }
</style>
""", unsafe_allow_html=True)


# load yolo model once and cache it
@st.cache_resource
def load_model():
    m = YOLO("yolov8n.pt")
    return m

model = load_model()


# try website first, fall back to local file for logo
@st.cache_data
def load_logo():
    try:
        url = "https://www.iiit.ac.in/wp-content/uploads/2024/01/IIIT_H_logo.png"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert("RGBA")
    except Exception:
        pass
    local_path = os.path.join(os.path.dirname(__file__), "IIIT_Logo.png")
    if os.path.exists(local_path):
        return Image.open(local_path).convert("RGBA")
    return None


# ============================================================
#  COCO CLASS MAPPING
#  COCO dataset has 80 classes total (IDs 0 to 79).
#  We only pick 5 road vehicle classes for this project.
#
#  COCO ID | COCO Class Name | Our Label
#  --------|-----------------|---------------------------
#     0    | Person          | SKIP - not a vehicle
#     1    | Bicycle         | Bicycle
#     2    | Car             | Car
#     3    | Motorcycle      | Two-Wheelers
#     4    | Airplane        | SKIP - not a road vehicle
#     5    | Bus             | Bus
#     6    | Train           | SKIP - not a road vehicle
#     7    | Truck           | Truck
#     8    | Boat            | SKIP - not a road vehicle
#
#  Important Note and Major Limitations:
#
#  (1) Indian Vehicle Types Not Present in COCO Dataset:
#  YOLOv8n is pre-trained on COCO which has 80 classes. Our code only
#  picks 5 road vehicle classes: Bicycle, Car, Two-Wheeler, Bus, and Truck.
#  Anything below the confidence threshold is not detected at all, and
#  anything detected but outside our 5 classes is simply skipped.
#  Auto-Rickshaws, Small Tempos, Tempo Travellers, and other LCVs like
#  construction vehicles are not present in COCO, so the model either misses
#  them entirely or sometimes confuses them with a car or truck. Additionally,
#  since YOLOv8n is pre-trained on the COCO dataset which is largely based on
#  Western road conditions, it has limited exposure to Indian vehicle types,
#  Indian traffic patterns, and mixed road environments commonly seen in India.
#  They would not automatically go into an 'Others/Auto-rickshaws/LCVs'
#  category even if we defined one in our code - it would either remain zero
#  or get misclassified under one of the 5 existing classes. To properly handle
#  such vehicles, we would need to fine-tune YOLOv8 on a custom Indian traffic
#  dataset, but that is outside the scope of this project as it requires
#  significant effort for both dataset creation and model fine-tuning.
#
#  (2) Peak Count vs Cumulative Count:
#  Peak count shows the maximum vehicles visible at any one moment.
#  Cumulative count shows total detections across the full video. Both are
#  useful but neither gives exact unique vehicle count without a proper
#  object tracker.
# ============================================================

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Two-Wheelers",   # COCO label is "motorcycle"
    5: "Bus",
    7: "Truck",
    # 0=person, 4=airplane, 6=train, 8=boat are all skipped
}

ALL_CATS = ["Bicycle", "Car", "Two-Wheelers", "Bus", "Truck"]

# bounding box colors - BGR for OpenCV
CAT_COLORS = {
    "Bicycle":      (0, 200, 255),
    "Car":          (50, 200, 50),
    "Two-Wheelers": (0, 255, 180),
    "Bus":          (200, 100, 0),
    "Truck":        (200, 0, 100),
}

# chart colors - hex for plotly
CHART_COLORS = {
    "Bicycle":      "#00c8ff",
    "Car":          "#32c832",
    "Two-Wheelers": "#00ffb4",
    "Bus":          "#c86400",
    "Truck":        "#c80064",
}


# ============================================================
#  DETECTION AND TRACKING FUNCTIONS
# ============================================================

def detect_vehicles(frame, phase2, conf_thresh, iou_thresh, imgsz, max_det):
    # plain detection - no tracking, just count per frame
    results = model(
        frame,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False
    )[0]

    cat_counts = {c: 0 for c in ALL_CATS}
    total = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue

        label = VEHICLE_CLASSES[cls_id]
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        total += 1
        if phase2:
            cat_counts[label] += 1

        color = CAT_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {conf:.2f}" if phase2 else f"Vehicle {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return frame, total, cat_counts


def detect_vehicles_tracked(frame, phase2, conf_thresh, iou_thresh, imgsz, max_det):
    # tracking mode - uses bytetrack to assign unique IDs per vehicle
    # returns same structure as detect_vehicles plus a set of track IDs seen
    results = model.track(
        frame,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        max_det=max_det,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]

    cat_counts = {c: 0 for c in ALL_CATS}
    total = 0
    track_ids_this_frame = set()
    cat_track_ids = {c: set() for c in ALL_CATS}

    if results.boxes.id is None:
        return frame, total, cat_counts, track_ids_this_frame, cat_track_ids

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue

        label = VEHICLE_CLASSES[cls_id]
        conf  = float(box.conf[0])
        tid   = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        total += 1
        track_ids_this_frame.add(tid)
        if phase2:
            cat_counts[label] += 1
            cat_track_ids[label].add(tid)

        color = CAT_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} ID:{tid}" if phase2 else f"ID:{tid}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return frame, total, cat_counts, track_ids_this_frame, cat_track_ids


def process_image(img_array, phase2, conf_thresh, iou_thresh, imgsz, max_det):
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    annotated, total, cats = detect_vehicles(
        frame, phase2, conf_thresh, iou_thresh, imgsz, max_det
    )
    # for image, unique = total (single frame, no duplicates possible)
    unique_total = total
    unique_cats  = dict(cats)
    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return out_rgb, total, cats, unique_total, unique_cats


def process_video(video_path, phase2, conf_thresh, iou_thresh, imgsz, max_det, progress_bar, status_text):
    # sample about 2 frames per second to keep processing fast on CPU/Colab
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_secs = total_frames / fps if fps > 0 else 0
    sample_every = max(1, int(fps / 2))

    frame_results = []
    cumulative_cats  = {c: 0 for c in ALL_CATS}
    cumulative_total = 0

    # tracking - collect unique track IDs across all frames
    all_track_ids = set()
    cat_all_track_ids = {c: set() for c in ALL_CATS}

    f_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        f_idx += 1
        if f_idx % sample_every != 0:
            continue

        # use tracking-based detection for unique count
        ann, total, cats, frame_tids, frame_cat_tids = detect_vehicles_tracked(
            frame.copy(), phase2, conf_thresh, iou_thresh, imgsz, max_det
        )
        frame_results.append((total, cats, ann))

        # cumulative totals (same vehicle counted every frame it appears)
        cumulative_total += total
        for c in ALL_CATS:
            cumulative_cats[c] += cats[c]

        # unique tracking IDs seen so far across all frames
        all_track_ids.update(frame_tids)
        for c in ALL_CATS:
            cat_all_track_ids[c].update(frame_cat_tids[c])

        pct = min(f_idx / max(total_frames, 1), 1.0)
        progress_bar.progress(pct)
        status_text.text(f"Processing frame {f_idx}/{total_frames}...")

    cap.release()
    progress_bar.progress(1.0)
    status_text.text("Done!")

    empty_cats = {c: 0 for c in ALL_CATS}
    if not frame_results:
        return None, 0, empty_cats, empty_cats, 0, 0, empty_cats, fps, duration_secs

    # best frame = frame with most vehicles visible
    best = max(frame_results, key=lambda x: x[0])
    peak_total = best[0]
    peak_cats  = best[1]
    best_frame = best[2]

    # unique counts from tracker IDs
    unique_total = len(all_track_ids)
    unique_cats  = {c: len(cat_all_track_ids[c]) for c in ALL_CATS}

    out_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    return out_rgb, peak_total, peak_cats, cumulative_cats, cumulative_total, unique_total, unique_cats, fps, duration_secs


def format_duration(secs):
    secs = int(secs)
    if secs < 60:
        return f"{secs} seconds"
    mins = secs // 60
    remaining = secs % 60
    return f"{mins} min {remaining} sec"


def get_file_size_mib(uploaded_file):
    uploaded_file.seek(0, 2)
    size_bytes = uploaded_file.tell()
    uploaded_file.seek(0)
    return size_bytes / (1024 * 1024)


# ============================================================
#  PLOTLY CHART HELPERS
# ============================================================

def make_bar_chart(cat_counts, title, color_override=None):
    # horizontal bar chart with count and percentage, interactive hover
    counts = [cat_counts.get(c, 0) for c in ALL_CATS]
    total  = sum(counts)
    pcts   = [round(v / total * 100, 1) if total > 0 else 0.0 for v in counts]
    colors = [color_override or CHART_COLORS[c] for c in ALL_CATS]

    hover_texts = [
        f"{ALL_CATS[i]}<br>Count: {counts[i]}<br>Percentage: {pcts[i]}%"
        for i in range(len(ALL_CATS))
    ]

    fig = go.Figure(go.Bar(
        x=counts,
        y=ALL_CATS,
        orientation="h",
        marker_color=colors,
        text=[f"{counts[i]}  ({pcts[i]}%)" for i in range(len(ALL_CATS))],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f5c518", size=13)),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(color="#e8e8e8"),
        xaxis=dict(title="Count", gridcolor="#333", zerolinecolor="#555"),
        yaxis=dict(title="", tickfont=dict(size=12)),
        margin=dict(l=10, r=60, t=40, b=30),
        height=260,
    )
    return fig


def make_unique_bar_chart(cat_counts, title):
    # same chart but green color to visually separate unique tracking results
    counts = [cat_counts.get(c, 0) for c in ALL_CATS]
    total  = sum(counts)
    pcts   = [round(v / total * 100, 1) if total > 0 else 0.0 for v in counts]

    hover_texts = [
        f"{ALL_CATS[i]}<br>Unique: {counts[i]}<br>Percentage: {pcts[i]}%"
        for i in range(len(ALL_CATS))
    ]

    fig = go.Figure(go.Bar(
        x=counts,
        y=ALL_CATS,
        orientation="h",
        marker_color="#00c875",
        text=[f"{counts[i]}  ({pcts[i]}%)" for i in range(len(ALL_CATS))],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#00c875", size=13)),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(color="#e8e8e8"),
        xaxis=dict(title="Unique Count", gridcolor="#333", zerolinecolor="#555"),
        yaxis=dict(title="", tickfont=dict(size=12)),
        margin=dict(l=10, r=60, t=40, b=30),
        height=260,
    )
    return fig


# ============================================================
#  CONTEXTUAL TIPS FOR DETECTION SETTINGS
# ============================================================

def conf_tip(val):
    if val <= 0.20:
        return f"Conf {val}: Very low - detects almost everything including many false positives. Use for testing only."
    elif val <= 0.30:
        return f"Conf {val}: Low - more detections but expect some wrong boxes mixed in."
    elif val <= 0.40:
        return f"Conf {val}: Medium - good balance between detections and accuracy. Commonly used in practice."
    elif val <= 0.55:
        return f"Conf {val}: Medium-high - fewer detections but more reliable results overall."
    elif val <= 0.70:
        return f"Conf {val}: High - only very confident detections shown. May miss some vehicles."
    else:
        return f"Conf {val}: Very high - extremely strict. Only obvious vehicles detected. Expect many misses."

def iou_tip(val):
    if val <= 0.25:
        return f"IoU {val}: Very strict NMS - overlapping boxes removed aggressively. Good for well-separated vehicles."
    elif val <= 0.40:
        return f"IoU {val}: Strict - reduces duplicate boxes well. Recommended for normal traffic scenes."
    elif val <= 0.55:
        return f"IoU {val}: Default range - balanced NMS. Works well in most traffic scenarios."
    elif val <= 0.70:
        return f"IoU {val}: Lenient - more boxes retained. Useful when vehicles are tightly packed together."
    else:
        return f"IoU {val}: Very lenient - many overlapping boxes kept. Risk of counting same vehicle twice."

def imgsz_tip(val):
    tips = {
        320: "Size 320: Fastest inference. Good for quick testing. May miss small or distant vehicles.",
        640: "Size 640: Default and recommended. Best balance of speed and accuracy for most videos.",
        1280: "Size 1280: Highest accuracy. Detects small vehicles better. Slower and needs more RAM."
    }
    return tips.get(val, f"Image size: {val}")

def maxdet_tip(val):
    if val <= 30:
        return f"Max {val}: Low cap. Fine for light traffic. Will miss vehicles if the scene is very busy."
    elif val <= 80:
        return f"Max {val}: Suitable for normal road traffic with moderate vehicle density."
    elif val <= 150:
        return f"Max {val}: Good for busy intersections or highway footage with many vehicles."
    else:
        return f"Max {val}: High cap. Handles very dense traffic scenes well. Slightly slower."


# ============================================================
#  UI - HEADER
# ============================================================

logo = load_logo()
if logo is not None:
    col_l, col_mid, col_r = st.columns([2, 1, 2])
    with col_mid:
        st.image(logo, width=160)

st.markdown('<div class="main-title">🚲 🚗 🛵 &nbsp; VEHICLE COUNTER &nbsp; 🚌 🚛</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">T5.4 SMAI (CS7.403) Assignment 3 - IIIT Hyderabad - YOLOv8n on COCO</div>', unsafe_allow_html=True)
st.markdown('<div class="prof-title">Course Coordinator : Prof. CV Jawahar</div>', unsafe_allow_html=True)
st.markdown('<div class="team-line">JAYSON JJ : 2025813005</div>', unsafe_allow_html=True)
st.markdown('<div class="team-line">PADMAJA Y : 2025813002 &nbsp;&nbsp; | &nbsp;&nbsp; GANGOTHRI CJ : 2025813004</div>', unsafe_allow_html=True)
st.markdown('<div class="team-name">Team: EarthLens</div>', unsafe_allow_html=True)


# ============================================================
#  UI - SIDEBAR
# ============================================================

with st.sidebar:
    with st.expander("Info - About this Project", expanded=False):
        st.markdown("""
**EarthLens - Vehicle Counter** is a project made as part of SMAI Assignment 3 at IIIT Hyderabad.
The app uses YOLOv8n (a pre-trained deep learning model) to detect and count vehicles from images or videos uploaded by the user.
No custom training was done - we are using the model as it is, already trained on the COCO dataset.

---

**Phase 1: Total Vehicles Count in Image/Video**

Upload any image or video and the app will detect all vehicles and show one single number as output.
Example output: "Total Vehicles Count: 12". This mode is simple and fast.

**Phase 2: Total Vehicles Count by Category in Image/Video**

Same as Phase 1 but also shows a breakdown by vehicle type.
Categories detected: Bicycle, Car, Two-Wheelers, Bus, Truck.
Useful for understanding what type of vehicles are present in the footage.

---

**Detection Settings**

Four settings can be adjusted to control how the model detects vehicles:
- Confidence Threshold: how sure the model must be before it counts a vehicle.
- IoU Threshold: controls how overlapping boxes are handled during Non-Maximum Suppression.
- Image Size: larger size gives better accuracy but runs slower.
- Max Detections: limits how many vehicles can be detected per frame.

---

**Important Note and Major Limitations:**

**(1) Indian Vehicle Types Not Present in COCO Dataset:**
YOLOv8n is pre-trained on COCO which has 80 classes. Our code only picks 5 road vehicle classes: Bicycle, Car, Two-Wheeler, Bus, and Truck. Anything below the confidence threshold is not detected at all, and anything detected but outside our 5 classes is simply skipped. Auto-Rickshaws, Small Tempos, Tempo Travellers, and other LCVs like construction vehicles are not present in COCO, so the model either misses them entirely or sometimes confuses them with a car or truck. Additionally, since YOLOv8n is pre-trained on the COCO dataset which is largely based on Western road conditions, it has limited exposure to Indian vehicle types, Indian traffic patterns, and mixed road environments commonly seen in India. They would not automatically go into an 'Others/Auto-rickshaws/LCVs' category even if we defined one in our code - it would either remain zero or get misclassified under one of the 5 existing classes. To properly handle such vehicles, we would need to fine-tune YOLOv8 on a custom Indian traffic dataset, but that is outside the scope of this project as it requires significant effort for both dataset creation and model fine-tuning.

**(2) Peak Count vs Cumulative Count:**
Peak count shows the maximum vehicles visible at any one moment. Cumulative count shows total detections across the full video. Both are useful but neither gives exact unique vehicle count without a proper object tracker.
        """)

    st.markdown("---")
    st.markdown("### Input")
    mode = st.radio("Input type", ["Image", "Video"])

    phase = st.radio(
        "Detection mode",
        [
            "Phase 1: Total Vehicles Count in Image/Video",
            "Phase 2: Total Vehicles Count by Category in Image/Video"
        ]
    )
    use_phase2 = ("Phase 2" in phase)

    st.markdown("---")
    st.markdown("### Detection Settings")

    conf_val = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
    st.markdown(f'<div class="tip-box">💡 {conf_tip(conf_val)}</div>', unsafe_allow_html=True)

    iou_val = st.slider("IoU Threshold (NMS)", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
    st.markdown(f'<div class="tip-box">💡 {iou_tip(iou_val)}</div>', unsafe_allow_html=True)

    imgsz_val = st.selectbox("Image Size (imgsz)", options=[320, 640, 1280], index=1)
    st.markdown(f'<div class="tip-box">💡 {imgsz_tip(imgsz_val)}</div>', unsafe_allow_html=True)

    maxdet_val = st.slider("Max Vehicles Detections per Frame", min_value=10, max_value=300, value=100, step=10)
    st.markdown(f'<div class="tip-box">💡 {maxdet_tip(maxdet_val)}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model:** YOLOv8n (COCO pretrained)")
    st.markdown("**Detects:** Bicycle, Car, Two-Wheelers, Bus, Truck")


# ============================================================
#  UI - MAIN AREA
#  Two-column layout. Left column has upload and right column
#  has preview. Results appear below in separate columns
#  depending on which mode and phase is selected.
# ============================================================

# -- shared state variables used across rows --
total        = 0
cats         = {c: 0 for c in ALL_CATS}
unique_total = 0
unique_cats  = {c: 0 for c in ALL_CATS}
peak_total   = 0
peak_cats    = {c: 0 for c in ALL_CATS}
cumul_total  = 0
cumul_cats   = {c: 0 for c in ALL_CATS}
unique_cumul_total = 0
unique_cumul_cats  = {c: 0 for c in ALL_CATS}

# row 1 - block A (upload) and block B (preview)
col_a, col_b = st.columns([1, 1.2], gap="large")

with col_a:
    st.markdown('<div class="section-hdr">Upload</div>', unsafe_allow_html=True)
    if mode == "Image":
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
        st.caption("Tip: Use a short clip (less than 30 sec) for faster processing on free hardware.")
    run_btn = st.button("Run Detection")

with col_b:
    st.markdown('<div class="section-hdr">Preview</div>', unsafe_allow_html=True)

    if run_btn and uploaded is not None:

        if mode == "Image":
            img_size_mib = get_file_size_mib(uploaded)
            img_size_kb  = img_size_mib * 1024
            img    = Image.open(uploaded).convert("RGB")
            img_np = np.array(img)
            img_w, img_h = img.size

            with st.spinner("Detecting vehicles..."):
                out_img, total, cats, unique_total, unique_cats = process_image(
                    img_np, use_phase2, conf_val, iou_val, imgsz_val, maxdet_val
                )

            st.image(out_img, caption="Annotated image", width='stretch')
            st.markdown(
                f'<div class="info-box">Resolution: {img_w} x {img_h} px &nbsp;|&nbsp; File size: {img_size_kb:.1f} KB</div>',
                unsafe_allow_html=True
            )

        else:  # video
            vid_size_mib = get_file_size_mib(uploaded)
            suffix = "." + uploaded.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            prog  = st.progress(0)
            s_txt = st.empty()

            out_img, peak_total, peak_cats, cumul_cats, cumul_total, unique_cumul_total, unique_cumul_cats, fps, duration_secs = process_video(
                tmp_path, use_phase2, conf_val, iou_val, imgsz_val, maxdet_val, prog, s_txt
            )
            os.unlink(tmp_path)

            if out_img is not None:
                st.image(out_img, caption="Best frame - most vehicles detected", width='stretch')

            dur_str = format_duration(duration_secs)
            st.markdown(
                f'<div class="info-box">Duration: {dur_str} &nbsp;|&nbsp; FPS: {fps:.1f} &nbsp;|&nbsp; File size: {vid_size_mib:.1f} MiB</div>',
                unsafe_allow_html=True
            )

            # cumulative for video phase1 is now rendered in its own row below
            # to properly align with peak count side by side

    elif run_btn and uploaded is None:
        st.warning("Please upload a file first!")
    else:
        st.info("Upload an image or video, then click Run Detection.")


# all rows below only render after detection is done
if run_btn and uploaded is not None:

    # image with phase 1 - show total count centered, then unique count below
    if mode == "Image" and not use_phase2:
        _, col_c, _ = st.columns([1, 2, 1])
        with col_c:
            st.markdown(
                f'<div class="count-box"><div class="count-label">Total Vehicles Count in Image</div>'
                f'<div class="count-num">{total}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        _, col_e, _ = st.columns([1, 2, 1])
        with col_e:
            st.markdown('<div class="section-hdr-green">Object Tracking - Unique Count</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="count-box-green"><div class="count-label">Total Unique Vehicles Count in Image</div>'
                f'<div class="count-num-green">{unique_total}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="track-note">For a single image, unique count equals total count since each vehicle appears only once.</div>',
                unsafe_allow_html=True
            )

    # image with phase 2 - category breakdown and chart side by side,
    # then unique tracking results below in same layout
    elif mode == "Image" and use_phase2:
        col_c, col_d = st.columns([1, 1.2], gap="large")
        with col_c:
            st.markdown('<div class="section-hdr">Category Breakdown</div>', unsafe_allow_html=True)
            for cat in ALL_CATS:
                v = cats.get(cat, 0)
                st.markdown(
                    f'<div class="cat-box"><span class="cat-name">{cat}</span>'
                    f'<span class="cat-val">{v}</span></div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f'<div class="count-box"><div class="count-label">Total Vehicles Count in Image</div>'
                f'<div class="count-num">{total}</div></div>',
                unsafe_allow_html=True
            )
        with col_d:
            st.markdown('<div class="section-hdr">Chart</div>', unsafe_allow_html=True)
            fig = make_bar_chart(cats, "Vehicles by Category")
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        col_e, col_f = st.columns([1, 1.2], gap="large")
        with col_e:
            st.markdown('<div class="section-hdr-green">Object Tracking - Unique Count by Category</div>', unsafe_allow_html=True)
            for cat in ALL_CATS:
                v = unique_cats.get(cat, 0)
                st.markdown(
                    f'<div class="cat-box-green"><span class="cat-name">{cat}</span>'
                    f'<span class="cat-val-green">{v}</span></div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f'<div class="count-box-green"><div class="count-label">Total Unique Vehicles Count in Image</div>'
                f'<div class="count-num-green">{unique_total}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="track-note">For a single image, unique count equals total count since each vehicle appears only once.</div>',
                unsafe_allow_html=True
            )
        with col_f:
            st.markdown('<div class="section-hdr-green">Unique Chart</div>', unsafe_allow_html=True)
            fig_u = make_unique_bar_chart(unique_cats, "Unique Vehicles by Category")
            st.plotly_chart(fig_u, width='stretch')

    # video with phase 1 - C1 and C2 in a fresh row below preview
    # both rendered together so they align properly side by side
    elif mode == "Video" and not use_phase2:
        col_c1, col_c2 = st.columns([1, 1.2], gap="large")
        with col_c1:
            st.markdown(
                f'<div class="count-box"><div class="count-label">Peak - Vehicles in Best Single Frame</div>'
                f'<div class="count-num">{peak_total}</div></div>',
                unsafe_allow_html=True
            )
        with col_c2:
            st.markdown('<div class="section-hdr">Cumulative - Total Vehicles Count Across All Frames</div>', unsafe_allow_html=True)
            st.caption("Same vehicle counted each frame it appears. Shows overall traffic volume.")
            st.markdown(
                f'<div class="count-box"><div class="count-label">Total Cumulative Vehicles Count</div>'
                f'<div class="count-num">{cumul_total}</div></div>',
                unsafe_allow_html=True
            )

        # unique tracking results below C1/C2
        st.markdown("---")
        col_e1, col_e2 = st.columns([1, 1], gap="large")
        with col_e1:
            st.markdown('<div class="section-hdr-green">Object Tracking - Unique Peak</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="count-box-green"><div class="count-label">Peak - Unique Vehicles in Best Single Frame</div>'
                f'<div class="count-num-green">{peak_total}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="track-note">Peak unique count equals peak total since tracking IDs in a single frame are always unique.</div>',
                unsafe_allow_html=True
            )
        with col_e2:
            st.markdown('<div class="section-hdr-green">Object Tracking - Unique Cumulative</div>', unsafe_allow_html=True)
            st.caption("Each vehicle counted only once across the entire video using ByteTrack IDs.")
            st.markdown(
                f'<div class="count-box-green"><div class="count-label">Total Unique Cumulative Vehicles Count</div>'
                f'<div class="count-num-green">{unique_cumul_total}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="track-note">Note: if a vehicle leaves and re-enters the frame, it may get a new ID and be counted again.</div>',
                unsafe_allow_html=True
            )

    # video with phase 2 - peak and cumulative categories side by side,
    # then charts, then unique tracking results with category breakdown
    elif mode == "Video" and use_phase2:
        # row 2
        col_c1, col_c2 = st.columns([1, 1], gap="large")
        with col_c1:
            st.markdown('<div class="section-hdr">Peak Frame - Category Breakdown</div>', unsafe_allow_html=True)
            st.caption("All numbers from the same frame, so they add up to peak total below.")
            for cat in ALL_CATS:
                v = peak_cats.get(cat, 0)
                st.markdown(
                    f'<div class="cat-box"><span class="cat-name">{cat}</span>'
                    f'<span class="cat-val">{v}</span></div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f'<div class="count-box"><div class="count-label">Peak - Vehicles in Best Single Frame</div>'
                f'<div class="count-num">{peak_total}</div></div>',
                unsafe_allow_html=True
            )
        with col_c2:
            st.markdown('<div class="section-hdr">Cumulative - Total Vehicles Count Across All Frames</div>', unsafe_allow_html=True)
            st.caption("Same vehicle counted each frame it appears. Shows overall traffic volume.")
            for cat in ALL_CATS:
                v = cumul_cats.get(cat, 0)
                st.markdown(
                    f'<div class="cat-box"><span class="cat-name">{cat}</span>'
                    f'<span class="cat-val">{v}</span></div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f'<div class="count-box"><div class="count-label">Total Cumulative Vehicles Count</div>'
                f'<div class="count-num">{cumul_total}</div></div>',
                unsafe_allow_html=True
            )

        # row 3 - charts
        col_d1, col_d2 = st.columns([1, 1], gap="large")
        with col_d1:
            fig1 = make_bar_chart(peak_cats, "Peak Frame - Vehicles by Category")
            st.plotly_chart(fig1, width='stretch')
        with col_d2:
            fig2 = make_bar_chart(cumul_cats, "Cumulative - Vehicles by Category")
            st.plotly_chart(fig2, width='stretch')

        st.markdown("---")

        # row 4 - unique peak and unique cumulative counts
        col_e1, col_e2 = st.columns([1, 1], gap="large")
        with col_e1:
            st.markdown('<div class="section-hdr-green">Object Tracking - Unique Peak</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="count-box-green"><div class="count-label">Peak - Unique Vehicles in Best Single Frame</div>'
                f'<div class="count-num-green">{peak_total}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="track-note">Peak unique equals peak total since tracking IDs in a single frame are always unique.</div>',
                unsafe_allow_html=True
            )
        with col_e2:
            st.markdown('<div class="section-hdr-green">Object Tracking - Unique Cumulative</div>', unsafe_allow_html=True)
            st.caption("Each vehicle counted only once across the entire video using ByteTrack IDs.")
            st.markdown(
                f'<div class="count-box-green"><div class="count-label">Total Unique Cumulative Vehicles Count</div>'
                f'<div class="count-num-green">{unique_cumul_total}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="track-note">Note: if a vehicle leaves and re-enters the frame, it may get a new ID and be counted again.</div>',
                unsafe_allow_html=True
            )

        # row 5 - unique category breakdown only, no duplicate count boxes
        # E1/E2 already showed the total unique counts above
        col_f1, col_f2 = st.columns([1, 1], gap="large")
        with col_f1:
            st.markdown('<div class="section-hdr-green">Peak Frame - Unique Category Breakdown</div>', unsafe_allow_html=True)
            st.caption("Category counts from the best frame using ByteTrack IDs.")
            for cat in ALL_CATS:
                v = peak_cats.get(cat, 0)
                st.markdown(
                    f'<div class="cat-box-green"><span class="cat-name">{cat}</span>'
                    f'<span class="cat-val-green">{v}</span></div>',
                    unsafe_allow_html=True
                )
            fig_f1 = make_unique_bar_chart(peak_cats, "Unique Peak - by Category")
            st.plotly_chart(fig_f1, width='stretch')
        with col_f2:
            st.markdown('<div class="section-hdr-green">Cumulative - Unique Vehicles Count with Category Breakdown</div>', unsafe_allow_html=True)
            st.caption("Unique vehicles seen across entire video, broken down by type.")
            for cat in ALL_CATS:
                v = unique_cumul_cats.get(cat, 0)
                st.markdown(
                    f'<div class="cat-box-green"><span class="cat-name">{cat}</span>'
                    f'<span class="cat-val-green">{v}</span></div>',
                    unsafe_allow_html=True
                )
            fig_f2 = make_unique_bar_chart(unique_cumul_cats, "Unique Cumulative - by Category")
            st.plotly_chart(fig_f2, width='stretch')
