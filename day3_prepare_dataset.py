"""Day 3: build a manageable, label-aware sample from a large DriverSVT CSV.

This script reads the raw CSV in chunks. It never loads the complete 9 GB file
into memory. It keeps a deterministic, class-aware sample and writes compact
outputs for feature engineering and baseline modelling.

Run from the project root:
  python .\\day3_prepare_dataset.py ".\\data\\raw\\DriverSVT\\DriverSVT.csv"

Optional:
  python .\\day3_prepare_dataset.py ".\\data\\raw\\DriverSVT\\DriverSVT.csv" --chunksize 50000 --outdir day3_working
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Initial sampling rates: retain more examples from uncommon classes.
# These are sampling fractions, not class weights or model settings.
SAMPLE_FRACTIONS = {
    "Normal": 0.01,
    "Camera off": 0.10,
    "Distraction": 0.10,
    "Unfastened belt": 0.10,
    "Drowsiness": 0.15,
    "No attention": 0.20,
    "Using phone": 0.35,
    "Eating": 1.00,
    "Smoking": 1.00,
}

# Keep only columns useful for the first sensor/state experiments.
# head_pose and deviceinfo are intentionally left out of the first baseline.
CANDIDATE_FEATURES = [
    "speed", "gforce",
    "acc_X", "acc_Y", "acc_Z",
    "accelerometer_data_X", "accelerometer_data_Y", "accelerometer_data_Z",
    "gyroscope_data_X", "gyroscope_data_Y", "gyroscope_data_Z",
    "perclos", "mouth_open_ratio",
]
TARGET = "dangerousstate"
GROUP = "userid"
TIME_COLUMNS = ["datetime", "datetimestart"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the raw DriverSVT CSV")
    parser.add_argument("--outdir", default="day3_working", help="Output directory")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Fraction of sampled driver IDs assigned to test (0-1)")
    args = parser.parse_args()

    source = Path(args.csv_path).expanduser()
    outdir = Path(args.outdir).expanduser()
    if not source.is_file():
        print(f"ERROR: CSV not found: {source}", file=sys.stderr)
        return 2
    if not 0 < args.test_size < 1:
        print("ERROR: --test-size must be between 0 and 1.", file=sys.stderr)
        return 2

    outdir.mkdir(parents=True, exist_ok=True)
    try:
        header = pd.read_csv(source, nrows=0, low_memory=False).columns.tolist()
    except Exception as exc:
        print(f"ERROR reading CSV header: {exc}", file=sys.stderr)
        return 1

    required = [TARGET, GROUP]
    missing_required = [c for c in required if c not in header]
    if missing_required:
        print(f"ERROR: required columns not found: {missing_required}", file=sys.stderr)
        return 1

    features = [c for c in CANDIDATE_FEATURES if c in header]
    time_cols = [c for c in TIME_COLUMNS if c in header]
    usecols = list(dict.fromkeys(required + features + time_cols))
    if not features:
        print("ERROR: none of the expected feature columns were found.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    sampled_chunks: list[pd.DataFrame] = []
    full_label_counts: dict[str, int] = {}
    selected_label_counts: dict[str, int] = {}
    total_rows = 0

    print(f"Input: {source}")
    print(f"Reading in chunks of {args.chunksize:,}; selected columns: {len(usecols)}")
    try:
        for chunk_no, chunk in enumerate(pd.read_csv(
            source, usecols=usecols, chunksize=args.chunksize, low_memory=False
        ), start=1):
            total_rows += len(chunk)
            labels = chunk[TARGET].fillna("<MISSING>").astype(str)
            for label, count in labels.value_counts(dropna=False).items():
                full_label_counts[label] = full_label_counts.get(label, 0) + int(count)

            # Use deterministic, label-aware random sampling within each chunk.
            random_values = rng.random(len(chunk))
            fractions = labels.map(SAMPLE_FRACTIONS).fillna(0.05).to_numpy(dtype=float)
            chosen = chunk.loc[random_values < fractions].copy()
            if not chosen.empty:
                sampled_chunks.append(chosen)
                for label, count in chosen[TARGET].fillna("<MISSING>").astype(str).value_counts().items():
                    selected_label_counts[label] = selected_label_counts.get(label, 0) + int(count)

            if chunk_no % 10 == 0:
                print(f"Scanned {total_rows:,} rows; sampled {sum(len(x) for x in sampled_chunks):,} so far...")
    except Exception as exc:
        print(f"ERROR while scanning CSV: {exc}", file=sys.stderr)
        return 1

    if not sampled_chunks:
        print("ERROR: no rows were sampled. Check target labels and sampling settings.", file=sys.stderr)
        return 1

    data = pd.concat(sampled_chunks, ignore_index=True)
    # Make target/group consistent and parse candidate signals as numeric.
    data[TARGET] = data[TARGET].fillna("<MISSING>").astype(str)
    data[GROUP] = data[GROUP].astype(str)
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Save compact sample and sampling diagnostics.
    sample_path = outdir / "sampled_driver_data.csv"
    data.to_csv(sample_path, index=False)
    label_rows = []
    for label in sorted(set(full_label_counts) | set(selected_label_counts)):
        total = full_label_counts.get(label, 0)
        kept = selected_label_counts.get(label, 0)
        label_rows.append({
            "label": label,
            "full_dataset_rows": total,
            "sampled_rows": kept,
            "full_dataset_percent": 100 * total / max(total_rows, 1),
            "sampled_percent": 100 * kept / max(len(data), 1),
            "sampling_fraction_observed": kept / total if total else np.nan,
        })
    pd.DataFrame(label_rows).sort_values("full_dataset_rows", ascending=False).to_csv(
        outdir / "sampling_summary.csv", index=False
    )

    missingness = pd.DataFrame({
        "column": features,
        "missing_count_sample": [int(data[c].isna().sum()) for c in features],
        "missing_percent_sample": [100 * float(data[c].isna().mean()) for c in features],
    }).sort_values("missing_percent_sample", ascending=False)
    missingness.to_csv(outdir / "feature_missingness.csv", index=False)

    # Deterministic driver-separated split. No driver appears in both partitions.
    # sklearn is not required for this step.
    driver_ids = sorted(data[GROUP].dropna().unique().tolist())
    split_rng = np.random.default_rng(args.seed)
    shuffled_ids = np.array(driver_ids, dtype=object)
    split_rng.shuffle(shuffled_ids)
    n_test = max(1, int(round(len(shuffled_ids) * args.test_size))) if len(shuffled_ids) > 1 else 0
    test_ids = set(shuffled_ids[:n_test].tolist())
    data["split"] = np.where(data[GROUP].isin(test_ids), "test", "train")
    data.to_csv(sample_path, index=False)
    split_summary = data.groupby(["split", TARGET], dropna=False).size().rename("rows").reset_index()
    split_summary.to_csv(outdir / "split_label_summary.csv", index=False)
    pd.DataFrame({"userid": driver_ids, "split": ["test" if uid in test_ids else "train" for uid in driver_ids]}).to_csv(
        outdir / "driver_split.csv", index=False
    )

    report = [
        "DAY 3 — WORKING DATASET PREPARATION",
        f"Source file: {source.resolve()}",
        f"Source size bytes: {source.stat().st_size:,}",
        f"Full rows scanned: {total_rows:,}",
        f"Sampled rows: {len(data):,}",
        f"Distinct drivers in sample: {data[GROUP].nunique():,}",
        f"Features retained: {', '.join(features)}",
        f"Target: {TARGET}",
        f"Group field: {GROUP}",
        f"Test drivers: {len(test_ids):,}",
        f"Train drivers: {len(driver_ids) - len(test_ids):,}",
        "Sampling is class-aware; sampled proportions do not represent natural prevalence.",
        "The split is by driver ID; verify each class is represented in both partitions before model training.",
        "No model has been trained by this script.",
    ]
    (outdir / "day3_preparation_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nDAY 3 PREPARATION COMPLETE")
    print(f"Full rows scanned: {total_rows:,}")
    print(f"Sampled rows: {len(data):,}")
    print(f"Drivers in sample: {data[GROUP].nunique():,}")
    print(f"Outputs: {outdir.resolve()}")
    print("Read day3_preparation_report.txt and sampling_summary.csv first.")
    print("NOTE: sampling changes class prevalence; use original full counts when reporting dataset distribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
