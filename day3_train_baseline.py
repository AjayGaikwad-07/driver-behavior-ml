"""Day 3: train a first driver-separated sensor-only baseline on DriverSVT.

Run from the project root:
  python .\\day3_train_baseline.py

Reads day3_working/sampled_driver_data.csv and uses its existing driver-level
train/test assignment. The model deliberately excludes perclos and mouth_open_ratio
for the first sensor-only experiment. It does not modify the input data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

TARGET = "dangerousstate"
GROUP = "userid"
FEATURES = ["speed", "gforce", "acc_X", "acc_Y", "acc_Z", "accelerometer_data_X", "accelerometer_data_Y", "accelerometer_data_Z", "gyroscope_data_X", "gyroscope_data_Y", "gyroscope_data_Z", "perclos", "mouth_open_ratio",]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="day3_working/sampled_driver_data.csv")
    parser.add_argument("--outdir", default="day3_baseline_results")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path, outdir = Path(args.input), Path(args.outdir)
    if not path.is_file():
        print(f"ERROR: input not found: {path.resolve()}", file=sys.stderr)
        return 2
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    required = [TARGET, GROUP, "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing required columns: {missing}", file=sys.stderr)
        return 2
    features = [c for c in FEATURES if c in df.columns]
    if not features:
        print("ERROR: none of the sensor features were found.", file=sys.stderr)
        return 2
    df = df[df[TARGET].notna() & df["split"].isin(["train", "test"])].copy()
    df[TARGET] = df[TARGET].astype(str)
    df[GROUP] = df[GROUP].astype(str)
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    train, test = df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        print("ERROR: train or test partition is empty.", file=sys.stderr)
        return 2
    overlap = set(train[GROUP].unique()) & set(test[GROUP].unique())
    if overlap:
        print(f"ERROR: {len(overlap)} driver IDs occur in both train and test.", file=sys.stderr)
        return 2
    train_classes, test_classes = set(train[TARGET].unique()), set(test[TARGET].unique())
    missing_test_classes = sorted(train_classes - test_classes)
    missing_train_classes = sorted(test_classes - train_classes)
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]
    print(f"Rows: train={len(train):,}, test={len(test):,}")
    print(f"Drivers: train={train[GROUP].nunique():,}, test={test[GROUP].nunique():,}")
    print(f"Features ({len(features)}): {', '.join(features)}")
    print("Training sensor-only Random Forest; this may take several minutes...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", RandomForestClassifier(n_estimators=args.trees, max_depth=args.max_depth, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed))])
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = sorted(train_classes | test_classes)
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(outdir / "classification_report.csv", index=True)
    pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels).to_csv(outdir / "confusion_matrix.csv", index=True)
    pd.DataFrame({"feature": features, "importance": model.named_steps["classifier"].feature_importances_}).sort_values("importance", ascending=False).to_csv(outdir / "feature_importance.csv", index=False)
    metrics = {"model": "RandomForestClassifier", "features": ", ".join(features), "train_rows": len(train), "test_rows": len(test), "train_drivers": train[GROUP].nunique(), "test_drivers": test[GROUP].nunique(), "driver_overlap": len(overlap), "train_classes_missing_from_test": "; ".join(missing_test_classes), "test_classes_missing_from_train": "; ".join(missing_train_classes), "dummy_accuracy": accuracy_score(y_test, dummy_pred), "dummy_balanced_accuracy": balanced_accuracy_score(y_test, dummy_pred), "dummy_macro_f1": f1_score(y_test, dummy_pred, average="macro", zero_division=0), "accuracy": accuracy_score(y_test, pred), "balanced_accuracy": balanced_accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0), "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0)}
    pd.DataFrame([metrics]).to_csv(outdir / "metrics_summary.csv", index=False)
    lines = ["DAY 3 — SENSOR-ONLY BASELINE RESULTS", "This is an initial experiment, not a validated safety system.", f"Input: {path.resolve()}", f"Training rows: {len(train):,}", f"Test rows: {len(test):,}", f"Training drivers: {train[GROUP].nunique():,}", f"Test drivers: {test[GROUP].nunique():,}", f"Driver overlap: {len(overlap)} (must be zero)", f"Features: {', '.join(features)}", f"Classes missing from test (present in train): {missing_test_classes}", f"Classes missing from train (present in test): {missing_train_classes}", "", "Majority-class baseline:", f"  Accuracy: {metrics['dummy_accuracy']:.4f}", f"  Balanced accuracy: {metrics['dummy_balanced_accuracy']:.4f}", f"  Macro-F1: {metrics['dummy_macro_f1']:.4f}", "Random Forest:", f"  Accuracy: {metrics['accuracy']:.4f}", f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}", f"  Macro-F1: {metrics['macro_f1']:.4f}", f"  Weighted-F1: {metrics['weighted_f1']:.4f}", "", "Interpretation caution: this is a driver-separated split, but label prevalence differs sharply across train/test.", "Inspect per-class precision/recall and the confusion matrix; do not judge by accuracy alone.", "Feature importance is descriptive, not causal, and does not prove generalization."]
    (outdir / "baseline_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nResults saved to: {outdir.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
