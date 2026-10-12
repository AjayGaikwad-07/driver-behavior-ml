"""
Day 3 — Create a second driver-separated, class-aware evaluation split.

This script preserves the existing sampled rows and writes a NEW file. It searches
StratifiedGroupKFold candidates and selects the split with the best combination of
class-distribution similarity and test-size similarity to the current test fraction.
No driver may appear in both partitions.

Run from project root:
  python .\\day3_make_group_stratified_split.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

INPUT = Path("day3_working/sampled_driver_data.csv")
OUTDIR = Path("day3_group_split")
TARGET = "dangerousstate"
GROUP = "userid"
SPLIT = "split"
N_SPLITS = 3
SEEDS = range(50)


def distribution(labels, classes):
    counts = labels.value_counts().reindex(classes, fill_value=0).astype(float)
    total = counts.sum()
    return counts / total if total else counts


def main():
    if not INPUT.is_file():
        print(f"ERROR: Could not find {INPUT.resolve()}", file=sys.stderr)
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {INPUT} ...")
    df = pd.read_csv(INPUT, low_memory=False)
    needed = [TARGET, GROUP]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        return 2
    df = df[df[TARGET].notna() & df[GROUP].notna()].copy()
    y = df[TARGET].astype(str)
    groups = df[GROUP].astype(str)
    classes = sorted(y.unique())
    old_test_fraction = (df[SPLIT].astype(str) == "test").mean() if SPLIT in df else 0.30
    # Use the original split fraction as a size target, not as labels for the new split.
    # StratifiedGroupKFold test folds are approximately one third of the groups.
    target_test_fraction = old_test_fraction
    best = None
    candidate_count = 0

    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(df, y, groups)):
            candidate_count += 1
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            gtr, gte = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
            if gtr & gte:
                continue
            p_all = distribution(y, classes)
            p_tr = distribution(ytr, classes)
            p_te = distribution(yte, classes)
            # Total variation distances: 0 means identical label proportions.
            train_tv = 0.5 * float(np.abs(p_tr - p_all).sum())
            test_tv = 0.5 * float(np.abs(p_te - p_all).sum())
            size_error = abs(len(test_idx) / len(df) - target_test_fraction)
            missing_train = int((ytr.value_counts().reindex(classes, fill_value=0) == 0).sum())
            missing_test = int((yte.value_counts().reindex(classes, fill_value=0) == 0).sum())
            # Strongly penalize a missing class; also balance distribution and size.
            score = train_tv + test_tv + 0.5 * size_error + 2.0 * (missing_train + missing_test)
            if best is None or score < best["score"]:
                best = {
                    "score": score, "seed": seed, "fold": fold,
                    "train_idx": train_idx, "test_idx": test_idx,
                    "train_tv": train_tv, "test_tv": test_tv,
                    "size_error": size_error, "missing_train": missing_train,
                    "missing_test": missing_test,
                }

    if best is None:
        print("ERROR: no valid split candidate found.", file=sys.stderr)
        return 2

    # Copy the sampled data and assign a fresh split column.
    result = df.copy()
    result[SPLIT] = "train"
    result.iloc[best["test_idx"], result.columns.get_loc(SPLIT)] = "test"
    # Save rows in original order. Original split is intentionally replaced only in this NEW file.
    output_file = OUTDIR / "sampled_driver_data_group_stratified.csv"
    result.to_csv(output_file, index=False)

    train = result[result[SPLIT] == "train"]
    test = result[result[SPLIT] == "test"]
    overlap = set(train[GROUP].astype(str)) & set(test[GROUP].astype(str))
    all_dist = distribution(result[TARGET].astype(str), classes)
    train_dist = distribution(train[TARGET].astype(str), classes)
    test_dist = distribution(test[TARGET].astype(str), classes)
    summary = pd.DataFrame({
        "label": classes,
        "full_sample_rows": result[TARGET].astype(str).value_counts().reindex(classes, fill_value=0).values,
        "full_sample_percent": (all_dist.values * 100),
        "train_rows": train[TARGET].astype(str).value_counts().reindex(classes, fill_value=0).values,
        "train_percent": (train_dist.values * 100),
        "test_rows": test[TARGET].astype(str).value_counts().reindex(classes, fill_value=0).values,
        "test_percent": (test_dist.values * 100),
    })
    summary.to_csv(OUTDIR / "label_distribution_comparison.csv", index=False)
    pd.DataFrame([{
        "candidate_splits_checked": candidate_count,
        "selected_seed": best["seed"],
        "selected_fold": best["fold"],
        "score_lower_is_better": best["score"],
        "rows_total": len(result),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_row_percent": len(train) / len(result) * 100,
        "test_row_percent": len(test) / len(result) * 100,
        "train_drivers": train[GROUP].nunique(),
        "test_drivers": test[GROUP].nunique(),
        "driver_overlap": len(overlap),
        "train_total_variation_from_sample": best["train_tv"],
        "test_total_variation_from_sample": best["test_tv"],
        "classes_missing_train": "; ".join(sorted(set(classes) - set(train[TARGET].astype(str).unique()))),
        "classes_missing_test": "; ".join(sorted(set(classes) - set(test[TARGET].astype(str).unique()))),
    }]).to_csv(OUTDIR / "split_audit.csv", index=False)

    print("DAY 3 — GROUP-STRATIFIED SPLIT")
    print(f"Candidates checked: {candidate_count}")
    print(f"Selected seed/fold: {best['seed']}/{best['fold']}")
    print(f"Rows: train={len(train):,}; test={len(test):,}")
    print(f"Drivers: train={train[GROUP].nunique():,}; test={test[GROUP].nunique():,}")
    print(f"Driver overlap: {len(overlap)}")
    print(f"Train/test TV distance from overall sample: {best['train_tv']:.4f}/{best['test_tv']:.4f}")
    print(f"Missing classes train/test: {best['missing_train']}/{best['missing_test']}")
    print(f"Saved: {output_file.resolve()}")
    print(f"Saved: {(OUTDIR / 'split_audit.csv').resolve()}")
    print(f"Saved: {(OUTDIR / 'label_distribution_comparison.csv').resolve()}")
    print("Note: this is a second evaluation split, not proof that all driver-level confounding is removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
