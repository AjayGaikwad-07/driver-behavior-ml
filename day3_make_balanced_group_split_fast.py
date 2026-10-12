"""Fast candidate driver-separated split search for DriverSVT.

Run from the project root:
  python .\day3_make_balanced_group_split_fast.py

This uses a compact driver-by-label count matrix, so it avoids repeatedly
processing all 580k rows for every candidate. It writes a new output folder
and leaves previous data/results unchanged.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

INPUT = Path("day3_working/sampled_driver_data.csv")
OUTDIR = Path("day3_balanced_group_split_fast")
TARGET, GROUP = "dangerousstate", "userid"
CANDIDATES, TEST_DRIVER_FRACTION, TARGET_TEST_ROW_FRACTION, SEED = 500, 0.20, 0.30, 2026


def main():
    if not INPUT.is_file():
        print(f"ERROR: input not found: {INPUT.resolve()}", file=sys.stderr)
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {INPUT} ...", flush=True)
    df = pd.read_csv(INPUT, low_memory=False)
    if any(c not in df.columns for c in [TARGET, GROUP]):
        print(f"ERROR: required columns {TARGET!r}, {GROUP!r} not found.", file=sys.stderr)
        return 2
    df = df[df[TARGET].notna() & df[GROUP].notna()].copy().reset_index(drop=True)
    y = df[TARGET].astype(str)
    groups = df[GROUP].astype(str)
    classes = sorted(y.unique())
    group_codes, group_names = pd.factorize(groups, sort=True)
    class_codes, class_names = pd.factorize(y, sort=True)
    n_groups, n_classes = len(group_names), len(class_names)

    # Summarize rows once into a small driver x label matrix.
    counts = np.zeros((n_groups, n_classes), dtype=np.int64)
    np.add.at(counts, (group_codes, class_codes), 1)
    total_class_counts = counts.sum(axis=0).astype(float)
    overall_p = total_class_counts / total_class_counts.sum()
    n_rows = len(df)
    print(f"Prepared summary for {n_groups} drivers and {n_classes} labels.", flush=True)
    print(f"Scoring {CANDIDATES} candidates...", flush=True)

    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31 - 1, size=CANDIDATES)
    best = None
    all_groups = np.arange(n_groups)
    n_test_groups = max(1, round(n_groups * TEST_DRIVER_FRACTION))
    for i, seed in enumerate(seeds, 1):
        local_rng = np.random.RandomState(int(seed))
        test_group_ids = local_rng.choice(all_groups, size=n_test_groups, replace=False)
        test_group_mask = np.zeros(n_groups, dtype=bool)
        test_group_mask[test_group_ids] = True
        test_counts = counts[test_group_mask].sum(axis=0).astype(float)
        train_counts = total_class_counts - test_counts
        n_test = test_counts.sum()
        n_train = train_counts.sum()
        if not n_train or not n_test:
            continue
        p_test, p_train = test_counts / n_test, train_counts / n_train
        tv_test = 0.5 * np.abs(p_test - overall_p).sum()
        tv_train = 0.5 * np.abs(p_train - overall_p).sum()
        row_fraction = n_test / n_rows
        driver_fraction = n_test_groups / n_groups
        missing = int((test_counts == 0).sum() + (train_counts == 0).sum())
        score = (tv_train + tv_test
                 + 1.5 * abs(row_fraction - TARGET_TEST_ROW_FRACTION)
                 + 1.5 * abs(driver_fraction - TEST_DRIVER_FRACTION)
                 + 2.0 * missing)
        if best is None or score < best["score"]:
            best = dict(score=score, seed=int(seed), test_group_mask=test_group_mask.copy(),
                        tv_train=tv_train, tv_test=tv_test,
                        missing_train=int((train_counts == 0).sum()),
                        missing_test=int((test_counts == 0).sum()))
        if i % 100 == 0:
            print(f"Checked {i}/{CANDIDATES} candidates...", flush=True)

    if best is None:
        print("ERROR: no valid candidate found.", file=sys.stderr)
        return 2
    test_mask = best["test_group_mask"][group_codes]
    result = df.copy()
    result["split"] = np.where(test_mask, "test", "train")
    train, test = result[result["split"] == "train"], result[result["split"] == "test"]
    overlap = set(train[GROUP].astype(str)) & set(test[GROUP].astype(str))

    overall_counts_s = result[TARGET].astype(str).value_counts().reindex(classes, fill_value=0)
    train_counts_s = train[TARGET].astype(str).value_counts().reindex(classes, fill_value=0)
    test_counts_s = test[TARGET].astype(str).value_counts().reindex(classes, fill_value=0)
    driver_all = result.groupby(TARGET)[GROUP].nunique().reindex(classes, fill_value=0)
    driver_train = train.groupby(TARGET)[GROUP].nunique().reindex(classes, fill_value=0)
    driver_test = test.groupby(TARGET)[GROUP].nunique().reindex(classes, fill_value=0)
    comparison = pd.DataFrame({
        "label": classes,
        "overall_rows": overall_counts_s.values,
        "overall_percent": overall_counts_s.values / len(result) * 100,
        "train_rows": train_counts_s.values,
        "train_percent": train_counts_s.values / max(len(train), 1) * 100,
        "test_rows": test_counts_s.values,
        "test_percent": test_counts_s.values / max(len(test), 1) * 100,
        "drivers_with_label_total": driver_all.values,
        "drivers_with_label_train": driver_train.values,
        "drivers_with_label_test": driver_test.values,
    })
    audit = pd.DataFrame([{
        "candidates_checked": CANDIDATES, "selected_seed": best["seed"],
        "score_lower_is_better": best["score"], "rows_total": len(result),
        "train_rows": len(train), "test_rows": len(test),
        "train_row_percent": len(train) / len(result) * 100,
        "test_row_percent": len(test) / len(result) * 100,
        "train_drivers": train[GROUP].nunique(), "test_drivers": test[GROUP].nunique(),
        "train_driver_percent": train[GROUP].nunique() / n_groups * 100,
        "test_driver_percent": test[GROUP].nunique() / n_groups * 100,
        "driver_overlap": len(overlap),
        "train_total_variation_from_overall": best["tv_train"],
        "test_total_variation_from_overall": best["tv_test"],
        "classes_missing_train": "; ".join(sorted(set(classes) - set(train[TARGET].astype(str).unique()))),
        "classes_missing_test": "; ".join(sorted(set(classes) - set(test[TARGET].astype(str).unique()))),
        "warning": "Heuristic split. Rare labels concentrated in very few drivers remain unreliable."
    }])
    out_csv = OUTDIR / "sampled_driver_data_balanced_group_split.csv"
    result.to_csv(out_csv, index=False)
    audit.to_csv(OUTDIR / "split_audit.csv", index=False)
    comparison.to_csv(OUTDIR / "label_distribution_comparison.csv", index=False)
    print("\nDAY 3 — FAST DRIVER-COUNT-AWARE GROUP SPLIT")
    print(f"Rows: train={len(train):,} ({len(train)/len(result):.1%}), test={len(test):,} ({len(test)/len(result):.1%})")
    print(f"Drivers: train={train[GROUP].nunique():,}, test={test[GROUP].nunique():,}")
    print(f"Driver overlap: {len(overlap)}")
    print(f"TV distance train/test: {best['tv_train']:.4f}/{best['tv_test']:.4f}")
    print(f"Missing classes train/test: {best['missing_train']}/{best['missing_test']}")
    print(f"Saved audit: {(OUTDIR / 'split_audit.csv').resolve()}")
    print(f"Saved comparison: {(OUTDIR / 'label_distribution_comparison.csv').resolve()}")
    print("Inspect the audit before training; this is a heuristic split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
