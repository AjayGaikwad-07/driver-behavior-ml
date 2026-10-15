
from pathlib import Path
import csv
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).parent
VIDEO = ROOT / "data" / "demo" / "driving_demo.mp4"
TASK = ROOT / "models" / "face_landmarker.task"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def ear(p, ids):
    return (
        distance(p[ids[1]], p[ids[5]]) +
        distance(p[ids[2]], p[ids[4]])
    ) / (2 * distance(p[ids[0]], p[ids[3]]) + 1e-9)


def mar(p):
    return distance(p[13], p[14]) / (distance(p[78], p[308]) + 1e-9)

if not VIDEO.exists() or not TASK.exists():
    raise FileNotFoundError("Video or face_landmarker.task not found.")

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(TASK)),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, round(fps / 5))  # sample ~5 frames per second

rows = []
closed_start = None
yawn_start = None
events = []
frame_no = 0

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % step:
            continue

        seconds = frame_no / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, int(seconds * 1000))

        if not result.face_landmarks:
            rows.append({
                "time_seconds": round(seconds, 2),
                "face_detected": False,
                "left_ear": "",
                "right_ear": "",
                "mouth_ratio": "",
            })
            continue

        p = result.face_landmarks[0]
        left = ear(p, [33, 160, 158, 133, 153, 144])
        right = ear(p, [362, 385, 387, 263, 373, 380])
        eye = (left + right) / 2
        mouth = mar(p)

        rows.append({
            "time_seconds": round(seconds, 2),
            "face_detected": True,
            "left_ear": round(left, 4),
            "right_ear": round(right, 4),
            "mouth_ratio": round(mouth, 4),
        })

        # Initial research thresholds; calibrate for this video.
        eyes_closed = eye < 0.20
        mouth_open = mouth > 0.55

        if eyes_closed:
            if closed_start is None:
                closed_start = seconds
        elif closed_start is not None:
            duration = seconds - closed_start
            if duration >= 1.5:
                events.append({
                    "event": "Suspected prolonged eye closure",
                    "start_seconds": round(closed_start, 2),
                    "end_seconds": round(seconds, 2),
                    "duration_seconds": round(duration, 2),
                })
            closed_start = None

        if mouth_open:
            if yawn_start is None:
                yawn_start = seconds
        elif yawn_start is not None:
            duration = seconds - yawn_start
            if duration >= 1.0:
                events.append({
                    "event": "Possible yawn / mouth opening",
                    "start_seconds": round(yawn_start, 2),
                    "end_seconds": round(seconds, 2),
                    "duration_seconds": round(duration, 2),
                })
            yawn_start = None

        if frame_no % (step * 50) == 0:
            print(f"Processed {seconds:.1f}s / {total / fps:.1f}s")

cap.release()

# Close any event still active at the end of the video.
end_time = frame_no / fps
for label, start in [
    ("Suspected prolonged eye closure", closed_start),
    ("Possible yawn / mouth opening", yawn_start),
]:
    if start is not None:
        duration = end_time - start
        threshold = 1.5 if "eye" in label else 1.0
        if duration >= threshold:
            events.append({
                "event": label,
                "start_seconds": round(start, 2),
                "end_seconds": round(end_time, 2),
                "duration_seconds": round(duration, 2),
            })

measurements = OUT / "face_measurements.csv"
event_file = OUT / "drowsiness_events.csv"

with measurements.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "time_seconds", "face_detected", "left_ear",
        "right_ear", "mouth_ratio"
    ])
    writer.writeheader()
    writer.writerows(rows)

with event_file.open("w", newline="", encoding="utf-8") as f:
    fields = ["event", "start_seconds", "end_seconds", "duration_seconds"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(events)

faces = sum(r["face_detected"] for r in rows)
print("\nANALYSIS COMPLETE")
print("Sampled frames:", len(rows))
print("Frames with face:", faces)
print("Events:", len(events))
print("Measurements:", measurements)
print("Event report:", event_file)
print("\nThese are preliminary estimates, not a validated sleep detector.")
