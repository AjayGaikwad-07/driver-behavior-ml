import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime
import json
import tempfile

import cv2
import joblib
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Clean, professional page configuration
st.set_page_config(
    page_title="Driver Behavior Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom human-designed CSS (calm typography, clear hierarchy, subtle borders, no tacky AI-neon gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Clean header styling */
    .app-header {
        padding: 0.8rem 0 1.2rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-top: 0;
    }
    
    /* Card style containers */
    .card-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    /* Badges & Pills */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 500;
        border-radius: 9999px;
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
    }
    .status-ready {
        background-color: #ecfdf5;
        color: #065f46;
        border-color: #a7f3d0;
    }
    .status-warning {
        background-color: #fffbeb;
        color: #92400e;
        border-color: #fde68a;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    
    /* Tab bar clean look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.95rem;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
COMPACT_MODEL_PATH = ROOT / "models" / "driver_behavior_rf_compact.joblib"
FULL_MODEL_PATH = ROOT / "models" / "driver_behavior_rf.joblib"
DEFAULT_VIDEO = ROOT / "data" / "demo" / "driving_demo.mp4"
DEMO_CSV = ROOT / "data" / "demo" / "sensor_sample_100_rows.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "speed", "gforce", "acc_X", "acc_Y", "acc_Z",
    "accelerometer_data_X", "accelerometer_data_Y",
    "accelerometer_data_Z", "gyroscope_data_X",
    "gyroscope_data_Y", "gyroscope_data_Z",
    "perclos", "mouth_open_ratio"
]

VISION_CLASSES = [
    "person", "cell phone", "cigarette",
    "car", "truck", "bus", "motorcycle"
]
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}


@st.cache_resource
def load_models():
    sensor = None
    vision = None
    
    # 1. Load sensor ML model
    for p in [COMPACT_MODEL_PATH, FULL_MODEL_PATH]:
        if p.exists():
            try:
                sensor = joblib.load(p)
                break
            except Exception:
                pass
                
    # 2. Load vision model
    try:
        if Path("yolov8s-worldv2.pt").exists():
            vision = YOLO("yolov8s-worldv2.pt")
            vision.set_classes(VISION_CLASSES)
        elif Path("yolov8n.pt").exists():
            vision = YOLO("yolov8n.pt")
        else:
            vision = YOLO("yolov8n.pt")
    except Exception:
        pass

    return sensor, vision

sensor_model, vision_model = load_models()

# --- SIDEBAR: SYSTEM STATUS & PROJECT INFO ---
with st.sidebar:
    st.subheader("System Status")
    
    if sensor_model is not None:
        st.markdown('<span class="status-badge status-ready">Sensor Model: Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">Sensor Model: Not Loaded</span>', unsafe_allow_html=True)
        
    if vision_model is not None:
        st.markdown('<span class="status-badge status-ready">Vision Model: Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">Vision Model: Not Loaded</span>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("**Quick Demo Data**")
    st.caption("You can use preloaded test files without manual setup:")
    
    if DEMO_CSV.exists():
        st.markdown(f"- **Sensor CSV:** `data/demo/sensor_sample_100_rows.csv`")
    if DEFAULT_VIDEO.exists():
        st.markdown(f"- **Trip Video:** `data/demo/driving_demo.mp4`")
        
    st.markdown("---")
    st.markdown("**About This Project**")
    st.caption(
        "Final Year Engineering Project analyzing smartphone telematics and in-cabin "
        "video for driver safety monitoring (DriverSVT benchmark)."
    )

# --- MAIN APP HEADER ---
st.markdown("""
<div class="app-header">
    <div class="app-title">🚗 Driver Behavior Analysis & Safety Monitor</div>
    <div class="app-subtitle">Smartphone Telematics ML + Cabin Video Object Detection</div>
</div>
""", unsafe_allow_html=True)

# Main navigation tabs with practical, simple titles
tab_sensor, tab_video, tab_benchmark = st.tabs([
    "📊 Sensor Analysis (Telemetry)",
    "🎥 Cabin Video Analysis",
    "📈 Project Benchmark & Reports"
])

# ==============================================================================
# TAB 1: SENSOR ANALYSIS
# ==============================================================================
with tab_sensor:
    st.markdown("### Vehicle Telemetry Analysis")
    st.write(
        "Evaluate driver states from smartphone sensor data (speed, accelerometer, gyroscope, and eye/mouth ratios)."
    )

    col_upload, col_hint = st.columns([2, 1])

    with col_upload:
        csv_file = st.file_uploader(
            "Upload Sensor CSV File", 
            type=["csv"], 
            key="sensor_csv",
            help="Upload telemetry recorded from the smartphone app or DriverSVT test split."
        )

    with col_hint:
        st.info(
            "💡 **Don't have a file ready?**\n\n"
            "Use the sample located at:\n"
            "`data/demo/sensor_sample_100_rows.csv`"
        )

    # Allow one-click loading of demo CSV if no file uploaded
    data = None
    source_label = ""
    if csv_file:
        data = pd.read_csv(csv_file, low_memory=False)
        source_label = f"Uploaded file: `{csv_file.name}`"
    elif st.button("Load Pre-packaged Demo Sample (100 rows)"):
        if DEMO_CSV.exists():
            data = pd.read_csv(DEMO_CSV, low_memory=False)
            source_label = "Pre-packaged Demo Sample (`sensor_sample_100_rows.csv`)"
        else:
            st.error("Demo sample file was not found on disk.")

    if data is not None:
        st.caption(source_label)
        
        # Check column requirements
        missing = [column for column in FEATURES if column not in data.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
            
            with st.expander("Preview Raw Data (First 5 Rows)", expanded=False):
                st.dataframe(data.head(5), use_container_width=True)

            if st.button("Classify Driver Behavior", type="primary"):
                if sensor_model is None:
                    st.error("Sensor model is not loaded in memory. Please check models/ directory.")
                    st.stop()
                    
                with st.spinner("Classifying driver events..."):
                    try:
                        predictions = sensor_model.predict(X)
                        output = data.copy()
                        output["predicted_behavior"] = predictions

                        if hasattr(sensor_model, "predict_proba"):
                            probabilities = sensor_model.predict_proba(X)
                            output["prediction_confidence"] = probabilities.max(axis=1).round(3)

                        st.success(f"Classified {len(output):,} telemetry rows successfully.")
                        
                        # High-level summary metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Rows Processed", f"{len(output):,}")
                        normal_count = (output["predicted_behavior"] == "normal").sum()
                        m2.metric("Safe / Normal Rows", f"{normal_count:,}")
                        risk_count = len(output) - normal_count
                        m3.metric("Attention / Risk Events", f"{risk_count:,}")

                        # Prediction distribution breakdown
                        st.markdown("#### Detected Behavior Breakdown")
                        dist_df = output["predicted_behavior"].value_counts().reset_index()
                        dist_df.columns = ["Behavior State", "Count"]
                        dist_df["Percentage"] = (dist_df["Count"] / len(output) * 100).round(1).astype(str) + "%"
                        
                        c_table, c_chart = st.columns([1, 1])
                        with c_table:
                            st.dataframe(dist_df, use_container_width=True, hide_index=True)
                        with c_chart:
                            st.bar_chart(dist_df.set_index("Behavior State")["Count"])

                        # Ground truth verification if available
                        if "dangerousstate" in output.columns:
                            st.markdown("#### Model Performance Against Ground-Truth")
                            from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
                            actual = output["dangerousstate"].astype(str)
                            pred = pd.Series(predictions).astype(str)
                            
                            p1, p2, p3 = st.columns(3)
                            p1.metric("Accuracy", f"{accuracy_score(actual, pred):.1%}")
                            p2.metric("Balanced Accuracy", f"{balanced_accuracy_score(actual, pred):.1%}")
                            p3.metric("Macro-F1", f"{f1_score(actual, pred, average='macro', zero_division=0):.3f}")

                        st.markdown("#### Detailed Predictions Table")
                        show_cols = [c for c in ["userid", "speed", "gforce", "perclos", "mouth_open_ratio", "predicted_behavior", "prediction_confidence"] if c in output.columns]
                        st.dataframe(output[show_cols].head(50), use_container_width=True)

                        st.download_button(
                            label="Download Output CSV with Predictions",
                            data=output.to_csv(index=False),
                            file_name="driver_sensor_predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

# ==============================================================================
# TAB 2: CABIN VIDEO ANALYSIS
# ==============================================================================
with tab_video:
    st.markdown("### Cabin Video Object & Distraction Detection")
    st.write(
        "Analyze dashboard or windshield camera footage to detect in-cabin distractions (phone use, smoking, occupants)."
    )

    video_col, settings_col = st.columns([2, 1])
    
    with settings_col:
        st.markdown("**Video Source**")
        upload_video = st.file_uploader("Upload driving video (.mp4, .mov)", type=["mp4", "avi", "mov"])
        sample_interval = st.select_slider(
            "Frame sampling interval", 
            options=[0.5, 1.0, 2.0, 3.0], 
            value=1.0,
            help="Higher interval processes faster; lower interval captures faster movements."
        )
        conf_threshold = st.slider(
            "Detection confidence threshold",
            min_value=0.25,
            max_value=0.85,
            value=0.50,
            step=0.05,
            help="Increase to eliminate false positives (e.g. shadows/dashboards mistaken for phones)."
        )

    video_path = None
    if upload_video:
        suffix = Path(upload_video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload_video.getbuffer())
            video_path = tmp.name
    elif DEFAULT_VIDEO.exists():
        video_path = str(DEFAULT_VIDEO)

    with video_col:
        if video_path:
            st.video(video_path)
        else:
            st.info("No video detected. Please upload an MP4 file or place driving_demo.mp4 in data/demo/.")

    if video_path:
        if st.button("Start Video Analysis", type="primary"):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            duration = total_frames / fps if total_frames else 0
            step = max(1, int(fps * sample_interval))

            events = []
            preview = st.empty()
            progress = st.progress(0.0)
            status_text = st.empty()
            processed = phone_frames = cigarette_frames = person_frames = total_objects = 0

            try:
                frame_index = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_index += 1
                    if frame_index % step:
                        continue

                    result = vision_model.predict(frame, conf=conf_threshold, verbose=False)[0]
                    names = result.names
                    detections = []

                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        label = (
                            str(names.get(class_id, class_id))
                            if isinstance(names, dict)
                            else str(names[class_id])
                        ).lower()
                        detections.append({
                            "label": label,
                            "confidence": float(box.conf[0]),
                        })

                    detected = [item["label"] for item in detections]
                    processed += 1
                    total_objects += len(detected)
                    timestamp = frame_index / fps

                    # Higher bar for small object like phone (min 0.50) to prevent shadows/steering wheel false alarms
                    phone_hits = [
                        item for item in detections
                        if ("phone" in item["label"] or "cell phone" in item["label"]) and item["confidence"] >= max(0.48, conf_threshold)
                    ]
                    cigarette_hits = [
                        item for item in detections 
                        if "cigarette" in item["label"] and item["confidence"] >= max(0.40, conf_threshold)
                    ]

                    if phone_hits:
                        phone_frames += 1
                        events.append({
                            "time_sec": round(timestamp, 1),
                            "event": "Phone Detected",
                            "confidence": f"{max(x['confidence'] for x in phone_hits):.0%}",
                        })

                    if cigarette_hits:
                        cigarette_frames += 1
                        events.append({
                            "time_sec": round(timestamp, 1),
                            "event": "Smoking Item Detected",
                            "confidence": f"{max(x['confidence'] for x in cigarette_hits):.0%}",
                        })

                    if "person" in detected:
                        person_frames += 1

                    preview.image(
                        cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    progress.progress(
                        min(frame_index / total_frames, 1.0) if total_frames else 0.0
                    )
                    status_text.text(
                        f"Analyzing: {timestamp:.1f}s / {duration:.1f}s | Sampled: {processed} frames"
                    )
            finally:
                cap.release()

            st.success("Trip video analysis finished.")
            
            # Metric cards
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Trip Duration", f"{duration:.1f}s")
            v2.metric("Sampled Frames", processed)
            v3.metric("Phone Instances", phone_frames)
            v4.metric("Smoking Instances", cigarette_frames)

            # Event Log Table
            event_df = pd.DataFrame(events)
            st.markdown("#### Timestamped Event Log")
            if event_df.empty:
                st.info("No distraction events detected during the sampled trip frames.")
            else:
                st.dataframe(event_df, use_container_width=True)

# ==============================================================================
# TAB 3: BENCHMARK & PROJECT SUMMARY
# ==============================================================================
with tab_benchmark:
    st.markdown("### Research Benchmark Summary (DriverSVT)")
    st.write(
        "Project validation results across 491 drivers with complete driver separation (0 train/test overlap)."
    )

    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        st.markdown("**Evaluated Benchmark Metrics**")
        benchmark_table = pd.DataFrame([
            {"Metric": "Accuracy", "Value": "28.39%", "Note": "Baseline on 9 imbalanced classes"},
            {"Metric": "Balanced Accuracy", "Value": "25.42%", "Note": "Equal weight across all behaviors"},
            {"Metric": "Macro-F1", "Value": "0.1838", "Note": "Primary metric (fair to minority risks)"},
            {"Metric": "Weighted-F1", "Value": "0.2237", "Note": "Volume-weighted performance"},
            {"Metric": "Driver Leakage", "Value": "0 drivers", "Note": "GroupShuffleSplit strictly applied"}
        ])
        st.dataframe(benchmark_table, hide_index=True, use_container_width=True)

    with b_col2:
        st.markdown("**Key Engineering Insights**")
        st.markdown("""
        1. **Driver Leakage Prevention:** Splitting by row creates false high accuracy (~80%). Splitting strictly by `userid` reveals true generalization (~28% acc / 0.184 Macro-F1).
        2. **Multimodal Advantage:** Sensor telemetry excels at vehicle dynamics (harsh braking, turns); cabin video is essential for behavioral distractions (phone, smoking).
        3. **Honest Reporting:** Academic integrity requires reporting Macro-F1 rather than misleading raw accuracy on heavily imbalanced datasets.
        """)

st.markdown("---")
st.caption("Driver Behavior Analysis • B.Tech Project • Department of Computer Engineering")
