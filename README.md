
# AI-Based Driver Behavior Analysis Using Smartphone Sensors

An ML research project exploring how smartphone sensor data can help identify driving behavior and potentially risky driving states.

## Project Overview

This project investigates machine learning techniques for analyzing driving behavior using smartphone-based sensing data. The long-term goal is to develop a system that can classify driving-related states, evaluate risk patterns, and summarize behavior at the journey level.

## Objectives

- Explore smartphone sensor and driving-state datasets.
- Perform exploratory data analysis (EDA) and data-quality checks.
- Develop baseline machine learning classifiers.
- Evaluate generalization to drivers unseen during training.
- Investigate class imbalance and appropriate evaluation metrics.
- Explore journey-level risk summaries as a future extension.

## Dataset

Initial dataset: [DriverSVT — Zenodo](https://zenodo.org/records/7385340)

The dataset contains smartphone telemetry and driving-state-related fields. Raw data is stored locally and is not included in this repository. Please consult the dataset documentation and usage restrictions before using it.

## Current Progress

- [x] Phase 1: Research paper review and project planning
- [x] Day 2: Full-dataset EDA scan
- [ ] Day 3: Feature engineering and baseline model
- [ ] Day 4: Advanced experiments and driver-separated evaluation
- [ ] Day 5: Risk analysis and prototype
- [ ] Day 6: Testing, documentation, and presentation

## Initial Technical Stack

- Python
- Pandas and NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Git and GitHub

## Evaluation Plan

The evaluation will consider macro-F1, per-class precision and recall, balanced accuracy, and a confusion matrix. Driver-separated evaluation will be investigated to reduce leakage between training and test data.

No model performance claims are made until experiments have been run and verified.

## Repository Structure

```text
driver-behavior-ai/
├── data/
│   └── raw/                 # Local data; not committed
├── reports/                 # Research and experiment reports
├── src/                     # ML implementation
├── day2_driver_dataset_eda.py
├── .gitignore
└── README.md
```

## Research and Responsible Use

This project is intended for research and educational purposes. It is not a validated safety-critical driver-monitoring system. Data privacy, dataset licensing, and limitations will be considered throughout development.
