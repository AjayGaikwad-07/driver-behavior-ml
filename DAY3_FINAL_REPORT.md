# Day 3 — Baseline Modeling and Evaluation Audit

## 1. Objective
Prepare a driver-separated dataset, train baseline classifiers, inspect
class-level performance, and document data-quality and evaluation limitations.

## 2. Dataset and split
- Dataset: DriverSVT
- Working sample: 580,806 rows
- Drivers represented: 491
- Training: 407,462 rows from 393 drivers
- Testing: 173,344 rows from 98 drivers
- Driver overlap: 0
- All nine classes appear in both partitions.

## 3. Latest baseline results
Model: Random Forest
Features: speed, gforce, acc_X, acc_Y, acc_Z,
accelerometer_data_X, accelerometer_data_Y, accelerometer_data_Z,
gyroscope_data_X, gyroscope_data_Y, gyroscope_data_Z, perclos,
mouth_open_ratio.

- Accuracy: 0.2665
- Balanced accuracy: 0.2687
- Macro-F1: 0.1876
- Weighted-F1: 0.2187

Majority-class baseline:
- Accuracy: 0.2740
- Balanced accuracy: 0.1111
- Macro-F1: 0.0478

The Random Forest has lower raw accuracy than the majority baseline,
but higher balanced accuracy and macro-F1.

## 4. Feature missingness
In the sampled working data:
- gyroscope_data_X/Y/Z: 0.2323% missing per column
- accelerometer_data_X/Y/Z: approximately 0.2266% missing per column
- Other audited columns: 0% missing

Missing-value handling should be documented and checked before final training.

## 5. Feature importance
Top features in the latest fitted Random Forest:
1. speed: 24.68%
2. accelerometer_data_X: 12.92%
3. mouth_open_ratio: 12.67%
4. accelerometer_data_Z: 10.96%
5. accelerometer_data_Y: 10.72%
6. perclos: 7.76%

Importance is model-specific and descriptive; it does not establish causation.

## 6. Class-level findings
- Drowsiness: recall approximately 45.47%, precision approximately 58.79%.
- No attention: recall approximately 85.02%, precision approximately 18.25%.
- Smoking: zero correct predictions.
- Using phone: recall approximately 3.74%.
- Unfastened belt: recall approximately 1.87%.

The model's performance varies substantially across classes.

## 7. Key evaluation limitation
The label distribution differs sharply between train and test. For example,
Camera off accounts for approximately 2.28% of training rows and 45.19%
of test rows. Some labels are represented by only two to seven drivers.
Driver separation prevents direct driver overlap, but does not remove
collection bias or establish robust generalization.

## 8. Potential label leakage and dataset provenance
The dataset documentation states that video-analysis models were used to
derive several driver-state labels. Investigate the relationship between
the target label and candidate features, collection sessions, and recording
conditions before making generalization claims. No confirmed leakage finding
is claimed at this stage.

## 9. Dataset use and privacy
The supplied dataset README restricts redistribution and public posting.
Do not commit the raw dataset, sampled data, or other restricted dataset
contents to a public repository. Keep local experiment outputs out of Git
unless each artifact is intentionally reviewed for safe publication.

## 10. Day 4 priorities
1. Inspect feature provenance and possible target leakage.
2. Define and document missing-value handling.
3. Review redundant sensor columns and feature construction.
4. Evaluate using multiple driver-separated splits or group-based validation.
5. Report macro-F1, balanced accuracy, per-class precision/recall, and
   confusion matrices; disclose limitations for rare classes.
6. Avoid selecting a split solely because it produces a higher score.

## Conclusion
The baseline pipeline runs and provides an initial evaluation on unseen
drivers. Results are limited by severe class imbalance, driver-level label
concentration, and train/test distribution shift. This is a research
prototype, not a validated driver-safety system.
