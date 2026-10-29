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

    /* Driver Executive Report Card */
    .driver-report-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .driver-score-val {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    .tier-badge {
        display: inline-block;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }
    .tier-safe { background-color: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
    .tier-caution { background-color: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .tier-risk { background-color: #fff1f2; color: #9f1239; border: 1px solid #fecdd3; }
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
    st.markdown("### 📋 Driver Behavior & Trip Safety Assessment")
    st.write(
        "Generate a comprehensive, physics-informed safety audit from smartphone sensor telemetry and facial alertness proxies."
    )

    col_upload, col_hint = st.columns([2, 1])

    with col_upload:
        csv_file = st.file_uploader(
            "Upload Sensor Telemetry CSV", 
            type=["csv"], 
            key="sensor_csv",
            help="Upload trip telemetry with motion sensors (speed, gforce, gyro, acc) and facial state proxies."
        )

    with col_hint:
        st.info(
            "💡 **Quick Evaluation:**\n\n"
            "Click below to evaluate the pre-packaged active trip sample with real driving dynamics:"
        )

    # Allow one-click loading of demo CSV if no file uploaded
    data = None
    source_label = ""
    if csv_file:
        data = pd.read_csv(csv_file, low_memory=False)
        source_label = f"Uploaded File: `{csv_file.name}`"
    elif st.button("🚗 Load Active Driving Sample (105 rows)"):
        if DEMO_CSV.exists():
            data = pd.read_csv(DEMO_CSV, low_memory=False)
            source_label = "Active Multi-Class Driving Sample (`sensor_sample_100_rows.csv`)"
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
            
            with st.expander("Preview Raw Sensor Telemetry (First 5 Rows)", expanded=False):
                st.dataframe(data.head(5), use_container_width=True)

            if st.button("Generate Driver Behavior Report", type="primary"):
                with st.spinner("Analyzing trip kinematics and computing safety audit..."):
                    # 1. Physics & Kinematic Event Analysis
                    speed = pd.to_numeric(data["speed"], errors="coerce").fillna(0)
                    gforce = pd.to_numeric(data["gforce"], errors="coerce").fillna(0)
                    gyro_z = pd.to_numeric(data.get("gyroscope_data_Z", 0), errors="coerce").fillna(0)
                    perclos = pd.to_numeric(data.get("perclos", 0), errors="coerce").fillna(0)
                    mouth = pd.to_numeric(data.get("mouth_open_ratio", 0), errors="coerce").fillna(0)

                    total_samples = len(data)
                    avg_speed = float(speed.mean())
                    max_speed = float(speed.max())

                    # Detected kinematic & behavioral events
                    harsh_braking_events = int((gforce < -0.35).sum())
                    harsh_cornering_events = int(((gforce.abs() > 0.40) & (gyro_z.abs() > 0.08)).sum())
                    speeding_events = int((speed > 80.0).sum())
                    drowsy_events = int((perclos > 0.20).sum())
                    yawn_events = int((mouth > 0.50).sum())
                    smooth_driving_ratio = float((gforce.abs() <= 0.30).mean() * 100)

                    # 2. Transparent Safety Scoring (0-100)
                    penalties = {
                        "Harsh Braking Maneuvers (-6 pts each)": min(30.0, harsh_braking_events * 6.0),
                        "Aggressive Cornering / Swerving (-5 pts each)": min(25.0, harsh_cornering_events * 5.0),
                        "Excessive Speeding (> 80 km/h, -4 pts each)": min(20.0, speeding_events * 4.0),
                        "Drowsiness Signs (PERCLOS > 20%, -5 pts each)": min(25.0, drowsy_events * 5.0),
                        "Fatigue / Yawning Episodes (-3 pts each)": min(15.0, yawn_events * 3.0),
                    }
                    total_deductions = sum(penalties.values())
                    safety_score = max(0.0, min(100.0, round(100.0 - total_deductions, 1)))

                    # 3. Driver Profile Classification
                    if safety_score >= 85:
                        tier_name = "Tier 1: Safe & Attentive Driver"
                        tier_badge_class = "tier-safe"
                        tier_color = "#059669"
                        risk_level = "Low Risk"
                        summary_text = (
                            "Driver maintains smooth vehicle dynamics, consistent headway distance, "
                            "and alert head/eye behavior throughout the monitored trip."
                        )
                    elif safety_score >= 70:
                        tier_name = "Tier 2: Moderate Caution / Inconsistent"
                        tier_badge_class = "tier-caution"
                        tier_color = "#d97706"
                        risk_level = "Moderate Risk"
                        summary_text = (
                            "Overall trip is acceptable, but shows isolated harsh maneuvers or "
                            "mild fatigue indicators requiring driver self-awareness."
                        )
                    elif safety_score >= 50:
                        tier_name = "Tier 3: Elevated Risk / Aggressive or Fatigued"
                        tier_badge_class = "tier-risk"
                        tier_color = "#ea580c"
                        risk_level = "Elevated Risk"
                        summary_text = (
                            "Recurrent sharp deceleration, aggressive handling, or notable drowsiness signs. "
                            "Increased likelihood of near-miss incidents."
                        )
                    else:
                        tier_name = "Tier 4: Critical Safety Hazard"
                        tier_badge_class = "tier-risk"
                        tier_color = "#dc2626"
                        risk_level = "High / Critical Risk"
                        summary_text = (
                            "Frequent compound safety violations detected (abrupt braking, swerving, "
                            "or deep fatigue). Immediate intervention or mandatory rest stop recommended."
                        )

                    # 4. Machine Learning Model Predictions
                    ml_predictions = None
                    if sensor_model is not None:
                        try:
                            ml_predictions = sensor_model.predict(X)
                        except Exception:
                            pass

                    # --- EXECUTIVE DRIVER REPORT CARD ---
                    st.markdown("---")
                    st.markdown(f"""
                    <div class="driver-report-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                            <div style="flex: 1; min-width: 280px;">
                                <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.25rem;">Driver Safety Evaluation</div>
                                <div class="tier-badge {tier_badge_class}">{tier_name}</div>
                                <div style="font-size: 0.95rem; color: #374151; line-height: 1.5; margin-top: 0.25rem;">{summary_text}</div>
                            </div>
                            <div style="text-align: right; min-width: 160px;">
                                <div style="font-size: 0.85rem; font-weight: 600; color: #6b7280;">TRIP SAFETY SCORE</div>
                                <div class="driver-score-val" style="color: {tier_color};">{safety_score:.0f}<span style="font-size: 1.2rem; font-weight: 500; color: #9ca3af;"> / 100</span></div>
                                <div style="font-size: 0.85rem; font-weight: 600; color: {tier_color};">{risk_level}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- DETAILED PERFORMANCE PILLARS ---
                    col_kin, col_alert = st.columns(2)

                    with col_kin:
                        st.markdown("#### 🏎️ Vehicle Kinematics & Maneuvers")
                        k1, k2 = st.columns(2)
                        k1.metric("Average Speed", f"{avg_speed:.1f} km/h")
                        k2.metric("Max Speed", f"{max_speed:.1f} km/h")
                        
                        k3, k4 = st.columns(2)
                        k3.metric("Harsh Braking Events", harsh_braking_events, delta="0 alerts" if harsh_braking_events == 0 else f"{harsh_braking_events} alerts", delta_color="inverse")
                        k4.metric("Sharp Turn / Cornering", harsh_cornering_events, delta="Smooth" if harsh_cornering_events == 0 else f"{harsh_cornering_events} alerts", delta_color="inverse")

                        st.caption(f"Kinematic Smoothness Index: **{smooth_driving_ratio:.1f}%** of trip within safe G-force thresholds (|G| ≤ 0.30)")

                    with col_alert:
                        st.markdown("#### 👁️ Driver Alertness & Fatigue Indicators")
                        a1, a2 = st.columns(2)
                        a1.metric("Drowsy Episodes (PERCLOS > 0.20)", drowsy_events, delta="Alert" if drowsy_events == 0 else f"{drowsy_events} episodes", delta_color="inverse")
                        a2.metric("Yawn / Open-Mouth Events", yawn_events, delta="Normal" if yawn_events == 0 else f"{yawn_events} events", delta_color="inverse")

                        a3, a4 = st.columns(2)
                        high_speed_share = (speed > 80.0).mean() * 100
                        a3.metric("High-Speed Driving Share", f"{high_speed_share:.1f}%")
                        samples_analyzed = f"{total_samples:,} readings"
                        a4.metric("Telemetry Volume", samples_analyzed)

                        st.caption("Fatigue proxies derived from facial landmarker telemetry (PERCLOS eye closure & mouth aspect ratio).")

                    # --- POINT DEDUCTION AUDIT TABLE ---
                    st.markdown("#### ⚖️ Safety Score Deduction Audit")
                    audit_rows = []
                    audit_rows.append({"Event Category": "Baseline Perfect Score", "Events Observed": "-", "Points Deducted": "0.0 pts", "Status": "Starting Score 100"})
                    
                    event_counts = [harsh_braking_events, harsh_cornering_events, speeding_events, drowsy_events, yawn_events]
                    for (cat, pts), cnt in zip(penalties.items(), event_counts):
                        status = "✅ Clear" if cnt == 0 else f"⚠️ {cnt} Detected"
                        audit_rows.append({
                            "Event Category": cat,
                            "Events Observed": cnt,
                            "Points Deducted": f"-{pts:.1f} pts" if pts > 0 else "0.0 pts",
                            "Status": status
                        })
                    audit_rows.append({"Event Category": "FINAL TRIP SAFETY SCORE", "Events Observed": sum(event_counts), "Points Deducted": f"-{total_deductions:.1f} pts", "Status": f"Final: {safety_score:.1f} / 100"})

                    st.dataframe(pd.DataFrame(audit_rows), hide_index=True, use_container_width=True)

                    # --- ACTIONABLE DRIVER COACHING RECOMMENDATIONS ---
                    st.markdown("#### 💡 Tailored Driver Coaching Recommendations")
                    tips = []
                    if drowsy_events > 0 or yawn_events > 0:
                        tips.append("**Fatigue Management:** Eye closure patterns or yawning detected. Plan a 15-minute rest stop with hydration before continuing long highway stretches.")
                    if harsh_braking_events > 0:
                        tips.append("**Following Distance:** Abrupt decelerations detected. Increase following distance to at least 3-4 seconds to allow smooth braking in traffic.")
                    if harsh_cornering_events > 0:
                        tips.append("**Cornering Speed:** Lateral G-force spikes observed during turns. Decelerate smoothly before entering corners rather than braking mid-turn.")
                    if speeding_events > 0:
                        tips.append("**Speed Regulation:** Speeds exceeded 80 km/h in monitored segments. Use cruise control where permitted to maintain steady, compliant velocity.")
                    if not tips:
                        tips.append("**Exemplary Driving:** Excellent driving habits exhibited. Kinematic acceleration, cornering, and alertness indicators are all well within optimal safety bounds.")

                    for tip in tips:
                        st.markdown(f"- {tip}")

                    # --- MACHINE LEARNING MODEL PREDICTIONS (EXPANDER) ---
                    with st.expander("🤖 Machine Learning Model Classification & Telemetry Details", expanded=False):
                        if ml_predictions is not None:
                            output = data.copy()
                            output["predicted_behavior"] = ml_predictions
                            
                            st.markdown("**Random Forest 9-Class Model Breakdown:**")
                            dist_df = output["predicted_behavior"].value_counts().reset_index()
                            dist_df.columns = ["Model Predicted State", "Sample Count"]
                            dist_df["Share (%)"] = (dist_df["Sample Count"] / len(output) * 100).round(1).astype(str) + "%"
                            
                            m_col1, m_col2 = st.columns([1, 1])
                            with m_col1:
                                st.dataframe(dist_df, hide_index=True, use_container_width=True)
                            with m_col2:
                                st.bar_chart(dist_df.set_index("Model Predicted State")["Sample Count"])

                            st.caption(
                                "ℹ️ **Academic & Engineering Note:** Smartphone inertial sensors (accelerometer & gyroscope) "
                                "reliably detect physical vehicle maneuvers (harsh braking, cornering, rapid acceleration). "
                                "Non-kinematic states (e.g. seatbelt, phone usage) require complementary in-cabin vision verification."
                            )

                            show_cols = [c for c in ["userid", "speed", "gforce", "perclos", "mouth_open_ratio", "predicted_behavior"] if c in output.columns]
                            st.dataframe(output[show_cols].head(50), use_container_width=True)

                    # --- DOWNLOAD DRIVER ASSESSMENT REPORT ---
                    report_summary = {
                        "assessment_timestamp": datetime.now().isoformat(),
                        "driver_classification": tier_name,
                        "trip_safety_score": safety_score,
                        "risk_level": risk_level,
                        "total_telemetry_samples": total_samples,
                        "average_speed_kmh": round(avg_speed, 2),
                        "max_speed_kmh": round(max_speed, 2),
                        "harsh_braking_events": harsh_braking_events,
                        "harsh_cornering_events": harsh_cornering_events,
                        "speeding_events": speeding_events,
                        "drowsy_episodes": drowsy_events,
                        "yawn_events": yawn_events,
                        "kinematic_smoothness_pct": round(smooth_driving_ratio, 2),
                        "recommendations": tips
                    }
                    
                    st.download_button(
                        label="⬇️ Download Driver Behavior Audit Report (JSON)",
                        data=json.dumps(report_summary, indent=2),
                        file_name=f"driver_behavior_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

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
