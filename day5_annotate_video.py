from pathlib import Path
import csv
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

root = Path(__file__).parent
video = root / "data" / "demo" / "driving_demo.mp4"
task = root / "models" / "face_landmarker.task"
out = root / "reports"
out.mkdir(exist_ok=True)

if not video.exists() or not task.exists():
    raise FileNotFoundError("Input video or Face Landmarker model is missing.")

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def ear(p, ids):
    return (dist(p[ids[1]], p[ids[5]]) +
            dist(p[ids[2]], p[ids[4]])) / (
            2 * dist(p[ids[0]], p[ids[3]]) + 1e-9)

def mar(p):
    return dist(p[13], p[14]) / (dist(p[78], p[308]) + 1e-9)

cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = out / "annotated_driver_demo.mp4"
writer = cv2.VideoWriter(
    str(output), cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)
if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

opts = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(task)),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

events = []
eye_start = None
mouth_start = None
frame_no = 0

try:
    with vision.FaceLandmarker.create_from_options(opts) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_no += 1
            sec = (frame_no - 1) / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )
            result = detector.detect_for_video(image, int(sec * 1000))

            status = "FACE NOT DETECTED"
            color = (0, 165, 255)

            if result.face_landmarks:
                p = result.face_landmarks[0]
                left = ear(p, [33, 160, 158, 133, 153, 144])
                right = ear(p, [362, 385, 387, 263, 373, 380])
                eye = (left + right) / 2
                mouth = mar(p)

                xs = [int(q.x * width) for q in p]
                ys = [int(q.y * height) for q in p]
                x1, x2 = max(0, min(xs)), min(width - 1, max(xs))
                y1, y2 = max(0, min(ys)), min(height - 1, max(ys))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                eye_closed = eye < 0.20
                mouth_open = mouth > 0.55

                if eye_closed:
                    if eye_start is None:
                        eye_start = sec
                elif eye_start is not None:
                    duration = sec - eye_start
                    if duration >= 1.5:
                        events.append({
                            "event": "Suspected prolonged eye closure",
                            "start_seconds": round(eye_start, 2),
                            "end_seconds": round(sec, 2),
                            "duration_seconds": round(duration, 2)
                        })
                    eye_start = None

                if mouth_open:
                    if mouth_start is None:
                        mouth_start = sec
                elif mouth_start is not None:
                    duration = sec - mouth_start
                    if duration >= 1.0:
                        events.append({
                            "event": "Possible yawn / mouth opening",
                            "start_seconds": round(mouth_start, 2),
                            "end_seconds": round(sec, 2),
                            "duration_seconds": round(duration, 2)
                        })
                    mouth_start = None

                if eye_closed:
                    status = "SUSPECTED EYE CLOSURE"
                    color = (0, 0, 255)
                elif mouth_open:
                    status = "POSSIBLE YAWN / MOUTH OPEN"
                    color = (0, 165, 255)
                else:
                    status = "FACE DETECTED"
                    color = (0, 255, 0)

                cv2.putText(frame, f"EAR: {eye:.3f}  MAR: {mouth:.3f}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
            else:
                # Do not count missing-face intervals as continuous events.
                eye_start = None
                mouth_start = None

            cv2.putText(frame, f"Time: {sec:.1f}s",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
            cv2.putText(frame, status, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, "Experimental prototype - verify alerts",
                (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

            writer.write(frame)

            if frame_no % max(1, int(fps * 10)) == 0:
                print(f"Processed {sec:.0f}s / {total / fps:.0f}s")

finally:
    cap.release()
    writer.release()

with (out / "annotated_video_events.csv").open(
        "w", newline="", encoding="utf-8") as f:
    fields = ["event", "start_seconds", "end_seconds", "duration_seconds"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(events)

print("\nANNOTATED VIDEO COMPLETE")
print("Video:", output)
print("Event report:", out / "annotated_video_events.csv")
print("Frames processed:", frame_no)
print("Candidate events:", len(events))
print("Note: threshold-based estimates, not validated drowsiness detection.")
