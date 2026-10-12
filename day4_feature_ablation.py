import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

TARGET = "dangerousstate"
GROUP = "userid"

MOTION_FEATURES = [
    "speed", "gforce",
    "acc_X", "acc_Y", "acc_Z",
    "accelerometer_data_X",
    "accelerometer_data_Y",
    "accelerometer_data_Z",
    "gyroscope_data_X",
    "gyroscope_data_Y",
    "gyroscope_data_Z",
]

STATE_FEATURES = ["perclos", "mouth_open_ratio"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="day3_balanced_group_split_fast/sampled_driver_data_balanced_group_split.csv",
    )
    parser.add_argument("--outdir", default="day4_feature_ablation")
    parser.add_argument("--trees", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path.resolve()}\n"
            "Check the --input path and try again."
        )

    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)

    required = {TARGET, GROUP}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    # Reproduce the existing driver-held-out split; do not split rows randomly.
    train = df[df["split"].astype(str).str.lower().eq("train")].copy()
    test = df[df["split"].astype(str).str.lower().eq("test")].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Could not find train/test values in the 'split' column. "
            "Inspect the CSV columns and split labels."
        )

    overlap = set(train[GROUP].dropna()) & set(test[GROUP].dropna())
    if overlap:
        raise ValueError(
            f"Driver leakage: {len(overlap)} userid values occur in both splits."
        )

    train = train.dropna(subset=[TARGET]).copy()
    test = test.dropna(subset=[TARGET]).copy()

    print(f"Train: {len(train):,} rows, {train[GROUP].nunique():,} drivers")
    print(f"Test:  {len(test):,} rows, {test[GROUP].nunique():,} drivers")
    print(f"Driver overlap: {len(overlap)}")
    print("Class labels:", sorted(df[TARGET].dropna().astype(str).unique()))

    feature_groups = {
        "motion_only": MOTION_FEATURES,
        "state_only": STATE_FEATURES,
        "combined": MOTION_FEATURES + STATE_FEATURES,
    }

    models = {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=18,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=args.trees,
            max_depth=18,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        ),
    }

    y_train = train[TARGET].astype(str)
    y_test = test[TARGET].astype(str)
    labels = sorted(df[TARGET].dropna().astype(str).unique())

    metrics = []
    all_reports = []

    for group_name, requested_features in feature_groups.items():
        features = [f for f in requested_features if f in df.columns]
        if not features:
            print(f"Skipping {group_name}: no requested features found.")
            continue

        print(f"\n=== Feature group: {group_name} ({len(features)} features) ===")

        X_train = train[features].apply(pd.to_numeric, errors="coerce")
        X_test = test[features].apply(pd.to_numeric, errors="coerce")

        for model_name, make_model in models.items():
            print(f"Training {model_name} ...")
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", make_model()),
            ])

            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            predictions = model.predict(X_test)
            prediction_output = pd.DataFrame({
                "userid": test[GROUP].values,
                "actual": y_test.values,
                "predicted": predictions,
                "model": model_name,
                "feature_group": group_name,
            })

            prediction_output.to_csv(
                outdir / f"{model_name}_{group_name}_predictions.csv",
                index=False,
            )
            predict_seconds = time.perf_counter() - start

            metrics.append({
                "model": model_name,
                "feature_group": group_name,
                "features": ", ".join(features),
                "n_features": len(features),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_drivers": train[GROUP].nunique(),
                "test_drivers": test[GROUP].nunique(),
                "driver_overlap": len(overlap),
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "macro_f1": f1_score(
                    y_test, predictions, labels=labels,
                    average="macro", zero_division=0
                ),
                "weighted_f1": f1_score(
                    y_test, predictions, labels=labels,
                    average="weighted", zero_division=0
                ),
                "train_seconds": round(train_seconds, 2),
                "predict_seconds": round(predict_seconds, 2),
            })

            report = classification_report(
                y_test, predictions, labels=labels,
                output_dict=True, zero_division=0
            )
            for label in labels:
                row = report[label]
                all_reports.append({
                    "model": model_name,
                    "feature_group": group_name,
                    "class": label,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1-score"],
                    "support": row["support"],
                })

            safe_name = f"{model_name}_{group_name}"
            pd.DataFrame(
                confusion_matrix(y_test, predictions, labels=labels),
                index=labels, columns=labels
            ).to_csv(outdir / f"{safe_name}_confusion_matrix.csv")

            print(
                f"{model_name} | accuracy={metrics[-1]['accuracy']:.4f} "
                f"| balanced_acc={metrics[-1]['balanced_accuracy']:.4f} "
                f"| macro-F1={metrics[-1]['macro_f1']:.4f} "
                f"| train={train_seconds:.1f}s"
            )

    if not metrics:
        raise ValueError("No experiments ran. Check the available feature columns.")

    metrics_df = pd.DataFrame(metrics).sort_values(
        "macro_f1", ascending=False
    )
    metrics_df.to_csv(outdir / "metrics_comparison.csv", index=False)
    pd.DataFrame(all_reports).to_csv(
        outdir / "per_class_metrics.csv", index=False
    )

    summary = {
        "input": str(path.resolve()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_drivers": int(train[GROUP].nunique()),
        "test_drivers": int(test[GROUP].nunique()),
        "driver_overlap": len(overlap),
        "labels": labels,
        "note": (
            "Exploratory comparison on one held-out-driver split. "
            "Not a validated safety system. Investigate target provenance, "
            "driver-level label concentration and class distribution shift. "
            "Row-level metrics may be dominated by long recordings."
        ),
    }
    (outdir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== FINAL COMPARISON (sorted by macro-F1) ===")
    print(metrics_df[
        ["model", "feature_group", "accuracy",
         "balanced_accuracy", "macro_f1", "weighted_f1",
         "train_seconds", "predict_seconds"]
    ].to_string(index=False))
    print(f"\nResults saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
