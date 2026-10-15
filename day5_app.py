
from pathlib import Path
from datetime import datetime
import json
import tempfile

import cv2
import joblib
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Driver AI", page_icon="🚗", layout="wide")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "driver_behavior_rf.joblib"
DEFAULT_VIDEO = ROOT / "data" / "demo" / "driving_demo.mp4"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "speed", "gforce", "acc_X", "acc_Y", "acc_Z",
    "accelerometer_data_X", "accelerometer_data_Y",
    "accelerometer_data_Z", "gyroscope_data_X",
    "gyroscope_data_Y", "gyroscope_data_Z",
    "perclos", "mouth_open_ratio"
]

st.title("🚗 AI Driver Behavior Monitoring")
st.caption("Day 5 research prototype | Video + Sensor CSV + Trip Report")

@st.cache_resource
def load_models():
    sensor = joblib.load(MODEL_PATH)
    vision = YOLO("yolo11n.pt")
    return sensor, vision

sensor_model, vision_model = load_models()

video_tab, sensor_tab, summary_tab = st.tabs(
    ["🎥 Video Analysis", "📡 Sensor Analysis", "📄 Reports"]
)

with video_tab:
    st.subheader("Recorded Trip Analysis")
    upload = st.file_uploader("Optional: upload another video", type=["mp4", "avi", "mov"])

    video_path = None
    if upload:
        suffix = Path(upload.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getbuffer())
            video_path = tmp.name
    elif DEFAULT_VIDEO.exists():
        video_path = str(DEFAULT_VIDEO)

    if video_path:
        st.video(video_path)
        interval = st.select_slider(
            "Analyze every (seconds)", options=[0.5, 1, 2, 3], value=1
        )
        if st.button("▶ Analyze Trip", type="primary"):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            duration = total_frames / fps if total_frames else 0
            step = max(1, int(fps * interval))
            events = []
            preview = st.empty()
            progress = st.progress(0.0)
            status = st.empty()
            processed = 0
            phone_frames = 0
            person_frames = 0
            total_objects = 0

            try:
                frame_index = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_index += 1
                    if frame_index % step:
                        continue

                    result = vision_model.predict(
                        frame, conf=0.35, verbose=False
                    )[0]
                    names = result.names
                    detected = [
                        names[int(box.cls[0])]
                        for box in result.boxes
                    ]
                    processed += 1
                    total_objects += len(detected)
                    timestamp = frame_index / fps

                    if "cell phone" in detected:
                        phone_frames += 1
                        events.append({
                            "timestamp_seconds": round(timestamp, 2),
                            "event": "Cell phone object detected",
                            "evidence": "YOLO object detection",
                        })
                    if "person" in detected:
                        person_frames += 1

                    if any(x in detected for x in ["car", "truck", "bus", "motorcycle"]):
                        events.append({
                            "timestamp_seconds": round(timestamp, 2),
                            "event": "Vehicle object detected",
                            "evidence": "YOLO object detection",
                        })

                    preview.image(
                        cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    progress.progress(
                        min(frame_index / total_frames, 1.0)
                        if total_frames else 0.0
                    )
                    status.write(
                        f"Time: {timestamp:.1f}s | Frames analyzed: {processed} "
                        f"| Objects in current frame: {len(detected)}"
                    )
            finally:
                cap.release()

            event_df = pd.DataFrame(events)
            if event_df.empty:
                event_df = pd.DataFrame(columns=[
                    "timestamp_seconds", "event", "evidence"
                ])

            report = {
                "analysis_time": datetime.now().isoformat(timespec="seconds"),
                "video": Path(video_path).name,
                "duration_seconds": round(duration, 2),
                "sample_interval_seconds": interval,
                "frames_analyzed": processed,
                "person_detection_frames": person_frames,
                "phone_detection_frames": phone_frames,
                "total_detected_objects_across_sampled_frames": total_objects,
                "note": (
                    "Object detections are not proof of distraction or drowsiness. "
                    "Speed and harsh braking require suitable telemetry."
                ),
            }

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            event_file = REPORT_DIR / f"video_events_{stamp}.csv"
            json_file = REPORT_DIR / f"video_report_{stamp}.json"
            event_df.to_csv(event_file, index=False)
            json_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

            st.success("Video analysis complete — report saved.")
            a, b, c = st.columns(3)
            a.metric("Duration", f"{duration:.1f} sec")
            b.metric("Frames analyzed", processed)
            c.metric("Phone detection frames", phone_frames)
            st.dataframe(event_df, use_container_width=True)
            st.download_button(
                "⬇ Download video event CSV",
                event_df.to_csv(index=False),
                file_name=event_file.name,
                mime="text/csv",
            )
            st.download_button(
                "⬇ Download video summary JSON",
                json.dumps(report, indent=2),
                file_name=json_file.name,
                mime="application/json",
            )
            st.warning(
                "This prototype detects general objects. It does not confirm "
                "sleep, drowsiness, driver phone use, speed or harsh braking."
            )
    else:
        st.info("Place a video at data/demo/driving_demo.mp4 or upload one above.")

with sensor_tab:
    st.subheader("Sensor CSV — trained Random Forest")
    st.write("Upload telemetry with the required feature columns.")
    st.caption("Model predictions are estimates, not validated safety decisions.")
    csv_file = st.file_uploader("Upload sensor CSV", type=["csv"], key="sensor_csv")

    if csv_file:
        data = pd.read_csv(csv_file, low_memory=False)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error("Missing feature columns: " + ", ".join(missing))
        else:
            X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
            if st.button("🧠 Predict Sensor States"):
                predictions = sensor_model.predict(X)
                output = data.copy()
                output["predicted_driver_state"] = predictions

                if hasattr(sensor_model, "predict_proba"):
                    probabilities = sensor_model.predict_proba(X)
                    output["model_confidence"] = probabilities.max(axis=1)

                st.success(f"Predictions generated for {len(output):,} rows.")
                st.dataframe(
                    output[["predicted_driver_state"]].head(30),
                    use_container_width=True
                )
                st.download_button(
                    "⬇ Download sensor predictions CSV",
                    output.to_csv(index=False),
                    file_name="sensor_predictions.csv",
                    mime="text/csv",
                )

                if "dangerousstate" in output.columns:
                    from sklearn.metrics import accuracy_score, f1_score
                    actual = output["dangerousstate"].astype(str)
                    pred = pd.Series(predictions).astype(str)
                    st.metric("Accuracy on uploaded labelled CSV",
                              f"{accuracy_score(actual, pred):.2%}")
                    st.metric("Macro-F1 on uploaded labelled CSV",
                              f"{f1_score(actual, pred, average='macro', zero_division=0):.3f}")
                    st.caption("These metrics are meaningful only if this is an independent labelled evaluation set.")

with summary_tab:
    st.subheader("Project Progress — Days 1–5")
    st.markdown("""
    - **Days 1–2:** Literature review and DriverSVT exploratory data analysis.
    - **Day 3:** Driver-separated Random Forest baseline.
    - **Day 4:** Feature ablation and model comparison.
    - **Day 5:** Recorded-video object detection and sensor prediction prototype.
    """)
    st.markdown("**Previously measured baseline metrics**")
    st.dataframe(pd.DataFrame([
        {"Metric": "Accuracy", "Value": "28.39%"},
        {"Metric": "Balanced accuracy", "Value": "25.42%"},
        {"Metric": "Macro-F1", "Value": "0.1838"},
        {"Metric": "Weighted-F1", "Value": "0.2237"},
        {"Metric": "Train / test driver overlap", "Value": "0"},
    ]), hide_index=True, use_container_width=True)
    st.info(
        "Video detections and sensor predictions are separate evidence streams. "
        "No fused driver-risk score is claimed. Drowsiness and harsh-braking "
        "detection require dedicated validated logic and suitable input data."
    )

st.divider()
st.caption("Experimental academic prototype — not a certified driver-safety system.")
