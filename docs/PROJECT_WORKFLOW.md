# Driver Behavior Monitoring — Project Workflow

## Project Objective
Build an experimental AI-based driver behavior monitoring prototype using
smartphone telemetry and video-based facial measurements.

## Day 1 — Research and Scope
- Reviewed four research papers.
- Defined the project scope and candidate driver behaviors.
- Identified the available DriverSVT dataset.

## Day 2 — Exploratory Data Analysis
- Inspected the dataset structure, features, missing values, and label distribution.
- Documented class imbalance and data preparation requirements.
- Raw dataset files remain local and are not uploaded to GitHub.

## Day 3 — Dataset Preparation and Baseline
- Prepared a working sample from the large telemetry dataset.
- Split data by driver to evaluate on unseen drivers.
- Trained a Random Forest baseline using motion and state-related features.
- Saved the trained model locally under models/.

## Day 4 — Model Evaluation
- Compared Random Forest and ExtraTrees.
- Evaluated combined, motion-only, and state-only feature groups.
- Used accuracy, balanced accuracy, macro-F1, weighted F1, and inference time.
- Kept drivers disjoint between training and test sets.

### Best combined-feature Random Forest result in the recorded comparison
- Accuracy: 0.2665
- Balanced accuracy: 0.2689
- Macro-F1: 0.1876
- Weighted F1: 0.2184
- Training time: 25.47 seconds
- Prediction time: 1.23 seconds

These are experimental results on one held-out-driver split, not production-level
driver safety performance.

## Day 5 — Video Analysis and Dashboard
- Analyzed a recorded driving demo using facial landmarks.
- Generated facial measurement and candidate-event CSV reports.
- Created an annotated video.
- Built a Streamlit dashboard for the prototype.

The facial heuristics and telemetry classifier are separate experimental
pipelines; they are not a fully integrated, validated safety system.

## Important Files
- README.md: project overview and usage.
- DAY3_FINAL_REPORT.md: Day 3 preparation and baseline results.
- DAY4_FINAL_REPORT.md: Day 4 evaluation and limitations.
- day2_driver_dataset_eda.py: exploratory analysis.
- day3_train_baseline.py: baseline training.
- day4_feature_ablation.py: feature/model comparisons.
- day4_driver_evaluation.py: driver-level evaluation.
- day5_face_analysis.py: facial measurement analysis.
- day5_annotate_video.py: annotated video generation.
- day5_app.py: Streamlit dashboard.

## Run the Dashboard
From the project root, activate your Python environment and run:

    python -m streamlit run .\day5_app.py

Install the project's required Python packages in that environment if needed.

## Data, Privacy, and Limitations
- Raw datasets, trained model binaries, and generated demo artifacts are kept
  local and excluded from Git where configured.
- Video-based events are heuristic candidates, not confirmed medical or
  behavioral diagnoses.
- The prototype must not be used as a real-world driving safety system.
