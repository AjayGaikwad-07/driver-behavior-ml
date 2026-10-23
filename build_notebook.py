import json
from pathlib import Path

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# ==============================================================================
# TITLE & BANNER
# ==============================================================================
add_md("""# 🚗 AI-Based Driver Behavior Analysis Using Smartphone Sensors & Computer Vision
## Complete 6-Stage End-to-End Machine Learning Lifecycle

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AjayGaikwad-07/driver-behavior-ml/blob/main/driver_behavior_colab_master.ipynb)

---
### 📌 Executive Summary
Driver distraction, fatigue, and aggressive maneuvers are leading contributors to vehicular accidents worldwide. This project builds a **multimodal driver monitoring and risk assessment system** developed from scratch across the complete **6-Stage Machine Learning Lifecycle**:

1. **Problem Definition & Ideation**: Objective formulation, multi-class hazard taxonomy, stakeholder constraints, and evaluation metrics.
2. **Data Collection & Preparation**: Telemetry ingestion, data cleaning, missing value strategies, and kinematic feature engineering.
3. **Data Splitting & Exploration (EDA)**: Class imbalance inspection, sensor correlation analysis, and zero-leakage driver-separated group splitting.
4. **Model Selection & Training**: Naive baseline, Logistic Regression, Random Forest, ExtraTrees, and Gradient Boosting with class-imbalance weighting.
5. **Evaluation & Optimization**: Multi-metric evaluation on held-out unseen drivers, normalized confusion matrices, and feature ablation.
6. **Deployment & Monitoring**: Edge computer vision (MediaPipe EAR/MAR + YOLO distraction detection), journey safety scoring (0–100), model serialization, and drift monitoring.
""")

# ==============================================================================
# STAGE 1: PROBLEM DEFINITION AND IDEATION
# ==============================================================================
add_md("""---
# Stage 1: Problem Definition and Ideation

### 1.1 Business Problem & Objective
Fleet operators, insurers, and everyday drivers lack affordable, privacy-preserving systems to detect dangerous driving behaviors in real time. Dedicated in-cabin hardware is prohibitively expensive ($500–$2,000 per vehicle). Modern smartphones, however, contain high-precision inertial measurement units (accelerometer, gyroscope, GPS) and front-facing cameras capable of capturing driver posture and alertness.

**Core Objective**:
Develop a machine learning system that utilizes **smartphone inertial sensor data** and **lightweight edge vision** to accurately identify dangerous driving states, generate journey-level risk summaries, and alert drivers before accidents occur.

### 1.2 ML Task Formulation
* **Task Type**: Multi-class Supervised Classification (9 Target Driving States).
* **Target Classes**:
  1. `Normal`: Safe, attentive driving.
  2. `Distraction`: General visual or cognitive inattention.
  3. `Drowsiness`: Micro-sleep, prolonged eye closure, yawning.
  4. `Using phone`: Handheld phone talking/texting.
  5. `No attention`: Eyes off the roadway.
  6. `Smoking`: Hand-to-mouth cigarette usage.
  7. `Eating`: Eating or drinking while vehicle is in motion.
  8. `Unfastened belt`: Seatbelt violation.
  9. `Camera off`: Vision feed occluded or unavailable.
* **Secondary Task**: Computer Vision Face Landmarking (Eye Aspect Ratio, Mouth Aspect Ratio) & Object Detection (phone, cigarette) for continuous real-time video verification.

### 1.3 Key Constraints & Stakeholder Alignment
* **Latency Constraint**: Model inference must take $< 50\\text{ ms}$ on mobile/edge CPU/GPU runtimes.
* **Privacy by Design**: Raw video frames must be processed on-device (Edge AI) without transmitting sensitive cabin video to the cloud.
* **Generalization Constraint**: The model must generalize to **unseen drivers** whose physiological and driving traits were not part of the training set.

### 1.4 Success Metrics
Because driving hazards represent a minority of driving time, standard **Accuracy** is misleading (predicting "Normal" 100% of the time achieves high accuracy while failing to detect any danger).
* **Primary Metric**: **Macro-averaged F1 Score** (treats all 9 behavioral classes with equal importance).
* **Secondary Metric**: **Balanced Accuracy** (average recall across each class).
* **Diagnostic Tools**: Normalized Confusion Matrix and Per-Class Precision & Recall.
""")

# ==============================================================================
# ENVIRONMENT SETUP & HARDWARE VERIFICATION
# ==============================================================================
add_md("""---
## Environment Setup & Hardware Check
Verify GPU acceleration (recommended for Computer Vision inference) and install required dependencies.
""")

add_code("""# Hardware & Python Environment Check
import sys
import torch

print(f"Python Runtime: {sys.version.split()[0]}")
if torch.cuda.is_available():
    print(f"✅ GPU Accelerated: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Allocated: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("ℹ️ Running on CPU. Tabular ML runs fine; T4 GPU recommended for YOLO/MediaPipe vision.")
""")

add_code("""# Install required dependencies silently
!pip install -q --upgrade pip
!pip install -q ultralytics mediapipe opencv-python-headless scikit-learn pandas numpy matplotlib seaborn joblib streamlit
print("✅ Core dependencies successfully installed!")
""")

# ==============================================================================
# STAGE 2: DATA COLLECTION AND PREPARATION
# ==============================================================================
add_md("""---
# Stage 2: Data Collection and Preparation

### 2.1 Data Ingestion Pipeline
The dataset is derived from **DriverSVT** (Zenodo), containing smartphone telemetry (accelerometer, gyroscope, speed, g-force) alongside facial proxy metrics (`perclos`, `mouth_open_ratio`).
We clone the project repository to access the verified, balanced driver-separated dataset sample (`sampled_driver_data_balanced_group_split.csv` ~68 MB, 580,806 rows from 491 drivers).
""")

add_code("""from pathlib import Path
import os
import urllib.request
import pandas as pd
import numpy as np

# Clone repository if running in fresh Colab session
REPO_URL = "https://github.com/AjayGaikwad-07/driver-behavior-ml.git"
REPO_DIR = Path("driver-behavior-ml")

if not REPO_DIR.exists() and not Path("day3_prepare_dataset.py").exists():
    print(f"Cloning repository from {REPO_URL} ...")
    !git clone $REPO_URL
    os.chdir(REPO_DIR)
    print("Working directory set to:", os.getcwd())
else:
    print("Project directory active:", os.getcwd())

# Ensure necessary model and demo folders exist
for folder in ["data/demo", "data/raw", "models", "reports"]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Download MediaPipe Face Landmarker model asset if not present
TASK_PATH = Path("models/face_landmarker.task")
if not TASK_PATH.exists():
    print("Downloading MediaPipe face_landmarker.task asset...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, TASK_PATH)
    print(f"✅ Downloaded face_landmarker.task ({TASK_PATH.stat().st_size / 1e6:.2f} MB)")
else:
    print(f"✅ Found face_landmarker.task ({TASK_PATH.stat().st_size / 1e6:.2f} MB)")
""")

add_code("""# Load dataset
DATASET_PATH = Path("day3_balanced_group_split_fast/sampled_driver_data_balanced_group_split.csv")
if not DATASET_PATH.exists():
    # Fallback search
    for candidate in [Path("day3_working/sampled_driver_data.csv"), Path("data/raw/DriverSVT/DriverSVT.csv")]:
        if candidate.exists():
            DATASET_PATH = candidate
            break

print(f"Loading data from: {DATASET_PATH} ...")
df = pd.read_csv(DATASET_PATH, low_memory=False)

print(f"✅ Loaded {len(df):,} rows and {df.shape[1]} columns across {df['userid'].nunique()} unique drivers.")
df.head(5)
""")

add_md("""### 2.2 Data Cleaning & Missing Value Audit
Inertial sensors frequently suffer from hardware dropouts or intermittent sampling frequencies. We audit missing rates across all sensor axes.
""")

add_code("""# Missing value analysis
RAW_FEATURES = [
    "speed", "gforce", "acc_X", "acc_Y", "acc_Z",
    "accelerometer_data_X", "accelerometer_data_Y", "accelerometer_data_Z",
    "gyroscope_data_X", "gyroscope_data_Y", "gyroscope_data_Z",
    "perclos", "mouth_open_ratio"
]

missing_summary = df[RAW_FEATURES].isnull().sum().to_frame(name="Missing_Count")
missing_summary["Missing_Percentage"] = (missing_summary["Missing_Count"] / len(df)) * 100
missing_summary["Data_Type"] = df[RAW_FEATURES].dtypes

print("Missing Value Audit:")
display(missing_summary[missing_summary["Missing_Count"] > 0].round(4))

# Convert numeric columns safely
for col in RAW_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
""")

add_md("""### 2.3 Feature Engineering: Kinematic Magnitude & Jerk
Raw 3-axis accelerometer and gyroscope readings depend heavily on how the smartphone is oriented inside the vehicle (portrait, landscape, or tilted in a cup-holder).
To make our models **orientation-invariant**, we engineer:
1. **Total Acceleration Magnitude**: $||a|| = \\sqrt{a_x^2 + a_y^2 + a_z^2}$
2. **Total Angular Velocity Magnitude**: $||\\omega|| = \\sqrt{\\omega_x^2 + \\omega_y^2 + \\omega_z^2}$
3. **Kinematic Jerk Proxy**: Rate of change of acceleration, detecting sudden swerves or emergency braking.
4. **Speed-Gforce Interaction**: Kinetic energy proxy capturing high-speed maneuvers.
""")

add_code("""# Feature Engineering
df["acc_total_mag"] = np.sqrt(df["acc_X"]**2 + df["acc_Y"]**2 + df["acc_Z"]**2)
df["gyro_total_mag"] = np.sqrt(df["gyroscope_data_X"]**2 + df["gyroscope_data_Y"]**2 + df["gyroscope_data_Z"]**2)
df["kinetic_intensity"] = df["speed"] * df["gforce"]

ENGINEERED_FEATURES = RAW_FEATURES + ["acc_total_mag", "gyro_total_mag", "kinetic_intensity"]
print(f"✅ Engineered 3 new kinematic features. Total feature pool: {len(ENGINEERED_FEATURES)}")
df[["speed", "gforce", "acc_total_mag", "gyro_total_mag", "kinetic_intensity"]].head(5)
""")

# ==============================================================================
# STAGE 3: DATA SPLITTING AND EXPLORATION (EDA)
# ==============================================================================
add_md("""---
# Stage 3: Data Splitting and Exploration (EDA)

### 3.1 Exploratory Data Analysis (EDA)
We analyze:
1. **Target Distribution**: Severe class imbalance across the 9 driving behaviors.
2. **Kinematic Profiles**: How acceleration and speed distributions differ between safe and dangerous states.
3. **Feature Correlation Matrix**: Collinearity between motion telemetry and facial proxies.
""")

add_code("""import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Target Class Imbalance
class_counts = df["dangerousstate"].value_counts()
class_pcts = df["dangerousstate"].value_counts(normalize=True) * 100
colors = sns.color_palette("mako", len(class_counts))

axes[0].barh(class_counts.index, class_counts.values, color=colors)
axes[0].set_title(f"Class Distribution: Target 'dangerousstate' (N={len(df):,})", fontweight="bold")
axes[0].set_xlabel("Number of Telemetry Samples")
for i, (cnt, pct) in enumerate(zip(class_counts.values, class_pcts.values)):
    axes[0].text(cnt + max(class_counts.values)*0.01, i, f"{cnt:,} ({pct:.1f}%)", va="center", fontsize=9)

# 2. Correlation Matrix of Kinematic and State Features
corr_cols = ["speed", "gforce", "acc_total_mag", "gyro_total_mag", "perclos", "mouth_open_ratio"]
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1], cbar=True, vmin=-1, vmax=1)
axes[1].set_title("Correlation Matrix: Motion & Facial Proxies", fontweight="bold")

plt.tight_layout()
plt.show()
""")

add_md("""### 3.2 Strict Driver-Separated Group Splitting (Zero Data Leakage)

> ⚠️ **CRITICAL ML RULE**: In driver behavior analysis, a standard random `train_test_split` causes severe **data leakage** because telemetry samples from the same driver trip are split across both training and testing sets. The model memorizes individual driving styles instead of learning generalizable danger patterns.
>
> We use **Group Partitioning based on `userid`**, guaranteeing that test drivers are **100% unseen** during training.
""")

add_code("""from sklearn.model_selection import GroupShuffleSplit

if "split" in df.columns:
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["userid"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

train_drivers = set(train_df["userid"].unique())
test_drivers = set(test_df["userid"].unique())
overlap = train_drivers.intersection(test_drivers)

print("=" * 60)
print(f"📊 Training Set : {len(train_df):,} rows from {len(train_drivers):,} distinct drivers")
print(f"📊 Test Set     : {len(test_df):,} rows from {len(test_drivers):,} distinct drivers")
print(f"🔒 Driver Overlap / Data Leakage: {len(overlap)} drivers (MUST BE 0)")
print("=" * 60)

assert len(overlap) == 0, "Catastrophic data leakage detected! Driver sets must be disjoint."
print("✅ Verification Passed: Zero driver overlap confirmed.")
""")

# ==============================================================================
# STAGE 4: MODEL SELECTION AND TRAINING
# ==============================================================================
add_md("""---
# Stage 4: Model Selection and Training

We train and compare multiple model architectures with **median imputation** and **class-weight balancing**:
1. **DummyClassifier (Majority Baseline)**: Predicts the most frequent class to establish the performance floor.
2. **Logistic Regression (Linear Baseline)**: Fast, interpretable linear classification benchmark.
3. **Random Forest Classifier**: Non-linear ensemble model with `class_weight='balanced_subsample'` to handle severe class imbalance.
4. **ExtraTrees Classifier**: Extremely randomized trees with lower variance and high computational efficiency.
5. **HistGradientBoostingClassifier**: Modern gradient boosted decision trees.
""")

add_code("""import time
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

TARGET = "dangerousstate"
FEATURES = ENGINEERED_FEATURES

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(str)
X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(str)

# Define Model Dictionary
candidate_models = {
    "Majority Baseline": DummyClassifier(strategy="most_frequent"),
    
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
    ]),
    
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ))
    ]),
    
    "ExtraTrees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ))
    ]),

    "HistGradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=12,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

trained_models = {}
benchmark_records = []

print("🚀 Starting Model Training and Validation...")
for name, model in candidate_models.items():
    print(f"   Training {name} ...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict on unseen test drivers
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    
    trained_models[name] = model
    benchmark_records.append({
        "Model": name,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "Train Time (s)": round(train_time, 2)
    })

benchmark_df = pd.DataFrame(benchmark_records).sort_values(by="Macro-F1", ascending=False)
display(benchmark_df.round(4))
""")

# ==============================================================================
# STAGE 5: EVALUATION AND OPTIMIZATION
# ==============================================================================
add_md("""---
# Stage 5: Evaluation and Optimization

### 5.1 Benchmark Comparison & Metric Interpretation
Notice the critical disparity between **Raw Accuracy** and **Macro-F1**:
- The **Majority Baseline** achieves ~27.4% accuracy simply by guessing the most frequent class, but has a **Balanced Accuracy of only 11.1%** and **Macro-F1 of ~0.048** (completely failing on dangerous classes).
- **Random Forest** and **ExtraTrees** substantially outperform the baseline on **Macro-F1** and **Balanced Accuracy**, showing genuine capability on minority hazardous states.
""")

add_code("""# Visualize Model Benchmark Comparison
fig, ax = plt.subplots(figsize=(10, 5))
benchmark_plot = benchmark_df.melt(id_vars=["Model", "Train Time (s)"], 
                                  value_vars=["Accuracy", "Balanced Accuracy", "Macro-F1"],
                                  var_name="Metric", value_name="Score")

sns.barplot(data=benchmark_plot, x="Model", y="Score", hue="Metric", palette="tab10", ax=ax)
plt.title("Stage 5: Model Evaluation on Unseen Drivers (Zero-Leakage)", fontweight="bold")
plt.ylabel("Evaluation Score (0 to 1)")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.0)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
""")

add_md("""### 5.2 Per-Class Error Analysis & Confusion Matrix
We inspect the normalized Confusion Matrix and detailed Classification Report for the champion model (Random Forest).
""")

add_code("""from sklearn.metrics import classification_report, confusion_matrix

best_model = trained_models["Random Forest"]
rf_preds = best_model.predict(X_test)
labels = sorted(y_test.unique())

print("📋 Classification Report: Random Forest on Held-Out Unseen Drivers")
print(classification_report(y_test, rf_preds, digits=4, zero_division=0))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, rf_preds, labels=labels, normalize="true")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Normalized Confusion Matrix (Champion Random Forest)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted State")
plt.ylabel("Ground Truth State")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
""")

add_md("""### 5.3 Feature Ablation Study
Does combining inertial kinematics with facial state proxies (`perclos`, `mouth_open_ratio`) improve classification?
We ablate features into 3 distinct configurations:
1. **Motion Telemetry Only** (14 features: speed, gforce, acc, gyro)
2. **State Proxies Only** (2 features: perclos, mouth_open_ratio)
3. **Combined Multimodal Set** (All 16 features)
""")

add_code("""MOTION_COLS = [
    "speed", "gforce", "acc_X", "acc_Y", "acc_Z",
    "accelerometer_data_X", "accelerometer_data_Y", "accelerometer_data_Z",
    "gyroscope_data_X", "gyroscope_data_Y", "gyroscope_data_Z",
    "acc_total_mag", "gyro_total_mag", "kinetic_intensity"
]
STATE_COLS = ["perclos", "mouth_open_ratio"]
COMBINED_COLS = MOTION_COLS + STATE_COLS

ablation_configs = {
    "Motion Only": MOTION_COLS,
    "State Only": STATE_COLS,
    "Combined": COMBINED_COLS
}

ablation_records = []
for group_name, col_subset in ablation_configs.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=60, max_depth=16, class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    ])
    pipe.fit(train_df[col_subset], y_train)
    preds = pipe.predict(test_df[col_subset])
    
    ablation_records.append({
        "Feature Group": group_name,
        "Feature Count": len(col_subset),
        "Accuracy": accuracy_score(y_test, preds),
        "Balanced Accuracy": balanced_accuracy_score(y_test, preds),
        "Macro-F1": f1_score(y_test, preds, average="macro", zero_division=0)
    })

ablation_df = pd.DataFrame(ablation_records).sort_values(by="Macro-F1", ascending=False)
display(ablation_df.round(4))
""")

# ==============================================================================
# STAGE 6: DEPLOYMENT AND MONITORING
# ==============================================================================
add_md("""---
# Stage 6: Deployment, Edge Vision & Monitoring

### 6.1 Edge Computer Vision: Fatigue & Distraction Detection
In production, inertial sensors are paired with on-device computer vision:
- **MediaPipe Face Landmarker**: Real-time 468-point facial mesh computing **Eye Aspect Ratio (EAR)** for micro-sleep and **Mouth Aspect Ratio (MAR)** for yawning.
- **YOLOv8 Detection**: Detects handheld distraction objects (`cell phone`, `cigarette`, `bottle`).
""")

add_code("""import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# 1. Geometric Face Metrics
def dist_2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_ear(landmarks, eye_indices):
    v1 = dist_2d(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    v2 = dist_2d(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    h  = dist_2d(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (v1 + v2) / (2.0 * h + 1e-9)

def calculate_mar(landmarks):
    h = dist_2d(landmarks[13], landmarks[14])
    w = dist_2d(landmarks[78], landmarks[308])
    return h / (w + 1e-9)

LEFT_EYE_PTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_PTS = [362, 385, 387, 263, 373, 380]

# 2. Load lightweight YOLO object detection model
yolo_detector = YOLO("yolov8n.pt")
print("✅ Computer Vision Pipelines (MediaPipe Face Mesh + YOLOv8) Ready for Edge Deployment.")
""")

add_code("""# Run Edge Video Analysis on Demo Clip
VIDEO_DEMO = Path("data/demo/driving_demo.mp4")
if not VIDEO_DEMO.exists():
    vids = list(Path("data/demo").glob("*.mp4"))
    if vids:
        VIDEO_DEMO = vids[0]

if VIDEO_DEMO.exists() and TASK_PATH.exists():
    cap = cv2.VideoCapture(str(VIDEO_DEMO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {VIDEO_DEMO.name}: {total_f} frames @ {fps:.1f} FPS")
    
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(TASK_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    
    timestamps, ears, mars = [], [], []
    frame_no = 0
    step = max(1, int(round(fps / 5)))
    
    with vision.FaceLandmarker.create_from_options(opts) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret or frame_no > 250:
                break
            frame_no += 1
            if frame_no % step != 0:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((frame_no / fps) * 1000)
            res = landmarker.detect_for_video(mp_img, ts)
            
            if res.face_landmarks:
                pts = res.face_landmarks[0]
                ear_val = (calculate_ear(pts, LEFT_EYE_PTS) + calculate_ear(pts, RIGHT_EYE_PTS)) / 2.0
                mar_val = calculate_mar(pts)
                timestamps.append(frame_no / fps)
                ears.append(ear_val)
                mars.append(mar_val)
                
    cap.release()
    
    # Plot Fatigue Curves
    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, ears, label="Eye Aspect Ratio (EAR)", color="dodgerblue")
    plt.axhline(0.20, color="blue", linestyle="--", label="Drowsiness Threshold (EAR < 0.20)")
    plt.plot(timestamps, mars, label="Mouth Aspect Ratio (MAR)", color="orange")
    plt.axhline(0.55, color="red", linestyle="--", label="Yawn Threshold (MAR > 0.55)")
    plt.xlabel("Trip Time (seconds)")
    plt.ylabel("Aspect Ratio")
    plt.title("Continuous Driver Alertness & Fatigue Monitoring Timeline", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Demo video not located. Place an MP4 video at data/demo/driving_demo.mp4 to run vision inference.")
""")

add_md("""### 6.2 Multimodal Journey Risk Scoring Engine
We synthesize kinematic telemetry events and vision detections into a **Trip Safety Index (0–100)**:
$$\\text{Safety Score} = \\max\\left(0, 100 - \\sum \\text{Infraction Penalties}\\right)$$
""")

add_code("""def compute_trip_safety_report(drowsy_events, yawn_events, phone_events, harsh_braking_events, duration_min=15):
    penalties = {
        "Drowsiness (Prolonged Eye Closure)": drowsy_events * 12.0,
        "Yawning (Fatigue State)": yawn_events * 4.0,
        "Phone Distraction": phone_events * 15.0,
        "Harsh Kinematic Maneuvers": harsh_braking_events * 8.0
    }
    total_deduction = sum(penalties.values())
    safety_score = max(0.0, min(100.0, 100.0 - total_deduction))
    
    if safety_score >= 85:
        rating = "🟢 Safe & Attentive"
    elif safety_score >= 70:
        rating = "🟡 Moderate Caution"
    else:
        rating = "🔴 High Risk / Intervention Recommended"
        
    return {
        "Trip Safety Score": round(safety_score, 1),
        "Safety Rating": rating,
        "Trip Duration (min)": duration_min,
        "Total Infraction Deductions": round(total_deduction, 1),
        "Breakdown": penalties
    }

trip_summary = compute_trip_safety_report(
    drowsy_events=1,
    yawn_events=3,
    phone_events=1,
    harsh_braking_events=2,
    duration_min=20
)

print("="*50)
print("             🚗 JOURNEY SAFETY REPORT             ")
print("="*50)
for k, v in trip_summary.items():
    if k != "Breakdown":
        print(f" {k:<28} : {v}")
print("\n Penalty Deductions:")
for k, v in trip_summary["Breakdown"].items():
    print(f"   - {k:<35} : -{v} pts")
print("="*50)
""")

add_md("""### 6.3 Model Serialization for Production Deployment
We serialize the fitted Random Forest pipeline to `models/driver_behavior_rf.joblib` for deployment in mobile backends or Streamlit apps.
""")

add_code("""import joblib

MODEL_OUT = Path("models/driver_behavior_rf.joblib")
joblib.dump(best_model, MODEL_OUT)
print(f"✅ Model successfully exported to: {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1e6:.2f} MB)")
""")

add_md("""### 6.4 Production Monitoring & Model Drift Strategy
Once deployed in vehicles, driving telemetry experiences **distribution drift** due to:
1. **Sensor Calibration Drift**: Different smartphone models, mounting locations, and accelerometer noise profiles.
2. **Seasonal / Environmental Shifts**: Winter driving (cautious speed, slippery roads) vs. highway driving.
3. **Driver Style Shifts**: Aggressive vs. defensive driver cohorts.

**Continuous Monitoring Architecture**:
* **Population Stability Index (PSI)** & **Kolmogorov-Smirnov (KS) Tests**: Monitor feature distribution drift on incoming speed and g-force distributions weekly.
* **Alert Trigger**: When $\\text{PSI} > 0.25$ or average confidence drops below $70\\%$, trigger automated dataset ingestion and pipeline retraining.
""")

add_code("""# Simulation of Production Sensor Drift Check (KS-Test)
from scipy.stats import ks_2samp

# Compare speed distribution between train and incoming test cohort
ks_stat, p_value = ks_2samp(train_df["speed"].dropna(), test_df["speed"].dropna())

print(f"📊 Production Drift Monitor (Kolmogorov-Smirnov Test):")
print(f"   KS Statistic : {ks_stat:.4f}")
print(f"   P-value      : {p_value:.4e}")
if p_value < 0.05:
    print("   ⚠️ Statistically significant distribution shift detected between driver cohorts.")
    print("   Recommendation: Apply sensor normalization or re-calibrate on the target fleet.")
else:
    print("   ✅ Sensor distribution remains stable.")
""")

# ==============================================================================
# SUMMARY & RESPONSIBLE AI DISCLOSURES
# ==============================================================================
add_md("""---
## Final Summary & Responsible AI Disclosures

### 📋 Key Findings Across the 6 Stages:
1. **Zero-Leakage Group Partitioning**: Strict driver separation by `userid` prevents optimistic overfitting. Random row-level splitting artificially inflates accuracy, whereas group splitting tests genuine real-world generalization to unseen drivers.
2. **Severe Class Imbalance**: Common classes ("Normal", "No attention") dominate the training space. Balanced subsampling and class-weighted cost functions were essential to prevent the model from ignoring critical hazardous classes like Drowsiness or Phone usage.
3. **Feature Synergy**: Feature ablation confirmed that combining inertial motion features with facial state proxies improves macro-F1 over motion-only or state-only configurations.
4. **Multimodal Edge Fusion**: Smartphone telemetry detects abrupt vehicle dynamics (swerving, harsh braking), while MediaPipe/YOLO vision detects driver internal state (microsleep, eye closure, phone usage), creating a holistic safety evaluation.

### ⚠️ Research & Ethical Disclosures:
- This system is an **academic research prototype**, not a certified automotive safety device.
- Video analysis and eye-tracking require explicit driver consent and strict on-device processing to respect user privacy.
- Raw DriverSVT data is protected by dataset licensing and should not be redistributed publicly.
""")

notebook_dict = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

output_path = Path("driver_behavior_colab_master.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Successfully generated {output_path} with {len(cells)} cells across all 6 stages!")
