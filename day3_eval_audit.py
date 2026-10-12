"""Day 3 evaluation audit for the DriverSVT working sample.

Run from the project root:
    python .\\day3_eval_audit.py

Reads day3_working/sampled_driver_data.csv and writes compact CSV/text reports.
This script does not train a model or modify the input dataset.
"""
from pathlib import Path
import sys
import pandas as pd

INPUT = Path("day3_working/sampled_driver_data.csv")
OUT = Path("day3_eval_audit")
TARGET = "dangerousstate"
GROUP = "userid"
SPLIT = "split"


def main():
    if not INPUT.is_file():
        print(f"ERROR: Cannot find {INPUT.resolve()}")
        print("Run this script from your driver-behavior project root.")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Reading {INPUT} ...")
    df = pd.read_csv(INPUT, low_memory=False)
    required = [TARGET, GROUP, SPLIT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Required columns missing: {missing}")
        print(f"Columns found: {list(df.columns)}")
        return 2

    df = df[df[TARGET].notna() & df[GROUP].notna() & df[SPLIT].notna()].copy()
    df[TARGET] = df[TARGET].astype(str)
    df[GROUP] = df[GROUP].astype(str)
    df[SPLIT] = df[SPLIT].astype(str)

    # Basic integrity and split checks
    train_ids = set(df.loc[df[SPLIT] == "train", GROUP].unique())
    test_ids = set(df.loc[df[SPLIT] == "test", GROUP].unique())
    overlap = sorted(train_ids & test_ids)
    unknown_splits = sorted(set(df[SPLIT]) - {"train", "test"})

    split_counts = df.groupby([SPLIT, TARGET]).size().rename("rows").reset_index()
    split_counts["within_split_percent"] = split_counts.groupby(SPLIT)["rows"].transform(lambda s: s / s.sum() * 100)
    split_counts.to_csv(OUT / "label_distribution_by_split.csv", index=False)

    driver_counts = df.groupby([GROUP, SPLIT]).size().rename("total_rows").reset_index()
    driver_label = df.groupby([GROUP, SPLIT, TARGET]).size().rename("label_rows").reset_index()
    driver_label = driver_label.merge(driver_counts, on=[GROUP, SPLIT], how="left")
    driver_label["within_driver_percent"] = driver_label["label_rows"] / driver_label["total_rows"] * 100
    driver_label.to_csv(OUT / "label_distribution_by_driver.csv", index=False)

    # Summarize how concentrated each label is across drivers.
    per_label_driver = df.groupby([TARGET, GROUP]).size().rename("rows").reset_index()
    concentration = per_label_driver.groupby(TARGET).agg(
        drivers_with_label=(GROUP, "nunique"),
        total_rows=("rows", "sum"),
        largest_driver_rows=("rows", "max"),
        median_rows_per_present_driver=("rows", "median"),
    ).reset_index()
    concentration["largest_driver_share_percent"] = concentration["largest_driver_rows"] / concentration["total_rows"] * 100
    concentration.to_csv(OUT / "label_driver_concentration.csv", index=False)

    numeric_cols = [c for c in df.columns if c not in {TARGET, GROUP, SPLIT}]
    missingness = pd.DataFrame({
        "column": numeric_cols,
        "missing_rows": [int(df[c].isna().sum()) for c in numeric_cols],
        "missing_percent": [float(df[c].isna().mean() * 100) for c in numeric_cols],
    }).sort_values("missing_percent", ascending=False)
    missingness.to_csv(OUT / "feature_missingness.csv", index=False)

    lines = [
        "DAY 3 — EVALUATION AUDIT",
        f"Input: {INPUT.resolve()}",
        f"Rows audited: {len(df):,}",
        f"Distinct drivers: {df[GROUP].nunique():,}",
        f"Train rows: {(df[SPLIT] == 'train').sum():,}",
        f"Test rows: {(df[SPLIT] == 'test').sum():,}",
        f"Train drivers: {len(train_ids):,}",
        f"Test drivers: {len(test_ids):,}",
        f"Driver overlap: {len(overlap)}",
        f"Unknown split values: {unknown_splits}",
        f"Labels in train: {df.loc[df[SPLIT] == 'train', TARGET].nunique()}",
        f"Labels in test: {df.loc[df[SPLIT] == 'test', TARGET].nunique()}",
        "",
        "Interpretation notes:",
        "- Compare label_distribution_by_split.csv to identify train/test prevalence shift.",
        "- label_driver_concentration.csv shows whether labels are concentrated in a small number of drivers.",
        "- label_distribution_by_driver.csv allows inspection of individual drivers and their label mix.",
        "- This audit describes the sampled data; it does not establish cause or prove leakage.",
        "- A driver-level split prevents driver-ID overlap, but session/time leakage and dataset collection effects still require documentation review.",
    ]
    (OUT / "audit_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReports saved in: {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
