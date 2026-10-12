"""Create a candidate driver-separated split with driver-count and row-count targets.
Run from project root: python .\\day3_make_balanced_group_split.py
Input: day3_working/sampled_driver_data.csv
Output: day3_balanced_group_split/ (new files; originals unchanged)
This is a heuristic split search; rare labels concentrated in few drivers remain unreliable.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

INPUT = Path('day3_working/sampled_driver_data.csv')
OUTDIR = Path('day3_balanced_group_split')
TARGET = 'dangerousstate'
GROUP = 'userid'
SPLIT = 'split'
CANDIDATES = 2000
TEST_DRIVER_FRACTION = 0.20
TARGET_TEST_ROW_FRACTION = 0.30
SEED = 2026


def proportions(s, classes):
    return s.value_counts().reindex(classes, fill_value=0).astype(float) / max(len(s), 1)


def main():
    if not INPUT.is_file():
        print(f'ERROR: input not found: {INPUT.resolve()}', file=sys.stderr)
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f'Reading {INPUT} ...')
    df = pd.read_csv(INPUT, low_memory=False)
    missing = [c for c in [TARGET, GROUP] if c not in df.columns]
    if missing:
        print(f'ERROR: missing columns: {missing}', file=sys.stderr)
        return 2
    df = df[df[TARGET].notna() & df[GROUP].notna()].copy().reset_index(drop=True)
    y = df[TARGET].astype(str)
    groups = df[GROUP].astype(str)
    classes = sorted(y.unique())
    unique_groups = groups.nunique()
    all_p = proportions(y, classes)
    best = None
    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31 - 1, size=CANDIDATES)
    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_DRIVER_FRACTION, random_state=int(seed))
        train_idx, test_idx = next(splitter.split(df, y, groups))
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        if train_groups & test_groups:
            continue
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        p_tr, p_te = proportions(ytr, classes), proportions(yte, classes)
        tv_tr = 0.5 * float(np.abs(p_tr - all_p).sum())
        tv_te = 0.5 * float(np.abs(p_te - all_p).sum())
        row_frac = len(test_idx) / len(df)
        driver_frac = len(test_groups) / unique_groups
        missing_train = int((ytr.value_counts().reindex(classes, fill_value=0) == 0).sum())
        missing_test = int((yte.value_counts().reindex(classes, fill_value=0) == 0).sum())
        row_error = abs(row_frac - TARGET_TEST_ROW_FRACTION)
        driver_error = abs(driver_frac - TEST_DRIVER_FRACTION)
        score = tv_tr + tv_te + 1.5 * row_error + 1.5 * driver_error + 2.0 * (missing_train + missing_test)
        if best is None or score < best['score']:
            best = dict(score=score, seed=int(seed), train_idx=train_idx, test_idx=test_idx,
                        tv_tr=tv_tr, tv_te=tv_te, row_frac=row_frac, driver_frac=driver_frac,
                        missing_train=missing_train, missing_test=missing_test)
    if best is None:
        print('ERROR: no valid split candidate found.', file=sys.stderr)
        return 2
    result = df.copy()
    result[SPLIT] = 'train'
    result.loc[best['test_idx'], SPLIT] = 'test'
    out_csv = OUTDIR / 'sampled_driver_data_balanced_group_split.csv'
    result.to_csv(out_csv, index=False)
    train, test = result[result[SPLIT] == 'train'], result[result[SPLIT] == 'test']
    overlap = set(train[GROUP].astype(str)) & set(test[GROUP].astype(str))
    p_all = proportions(result[TARGET].astype(str), classes)
    p_train = proportions(train[TARGET].astype(str), classes)
    p_test = proportions(test[TARGET].astype(str), classes)
    comparison = pd.DataFrame({
        'label': classes,
        'overall_rows': result[TARGET].astype(str).value_counts().reindex(classes, fill_value=0).values,
        'overall_percent': p_all.values * 100,
        'train_rows': train[TARGET].astype(str).value_counts().reindex(classes, fill_value=0).values,
        'train_percent': p_train.values * 100,
        'test_rows': test[TARGET].astype(str).value_counts().reindex(classes, fill_value=0).values,
        'test_percent': p_test.values * 100,
        'drivers_with_label_total': result.groupby(TARGET)[GROUP].nunique().reindex(classes, fill_value=0).values,
        'drivers_with_label_train': train.groupby(TARGET)[GROUP].nunique().reindex(classes, fill_value=0).values,
        'drivers_with_label_test': test.groupby(TARGET)[GROUP].nunique().reindex(classes, fill_value=0).values,
    })
    comparison.to_csv(OUTDIR / 'label_distribution_comparison.csv', index=False)
    audit = {
        'candidates_checked': CANDIDATES, 'selected_seed': best['seed'], 'score_lower_is_better': best['score'],
        'rows_total': len(result), 'train_rows': len(train), 'test_rows': len(test),
        'train_row_percent': len(train) / len(result) * 100, 'test_row_percent': len(test) / len(result) * 100,
        'train_drivers': train[GROUP].nunique(), 'test_drivers': test[GROUP].nunique(),
        'train_driver_percent': train[GROUP].nunique() / unique_groups * 100,
        'test_driver_percent': test[GROUP].nunique() / unique_groups * 100,
        'driver_overlap': len(overlap), 'train_total_variation_from_overall': best['tv_tr'],
        'test_total_variation_from_overall': best['tv_te'],
        'classes_missing_train': '; '.join(sorted(set(classes) - set(train[TARGET].astype(str).unique()))),
        'classes_missing_test': '; '.join(sorted(set(classes) - set(test[TARGET].astype(str).unique()))),
        'warning': 'Heuristic candidate split; inspect driver counts per class. Rare labels in very few drivers remain unreliable.'}
    pd.DataFrame([audit]).to_csv(OUTDIR / 'split_audit.csv', index=False)
    print('DAY 3 — DRIVER-COUNT-AWARE GROUP SPLIT')
    print(f'Candidates checked: {CANDIDATES}; selected seed: {best["seed"]}')
    print(f'Rows: train={len(train):,} ({len(train)/len(result):.1%}), test={len(test):,} ({len(test)/len(result):.1%})')
    print(f'Drivers: train={train[GROUP].nunique():,} ({train[GROUP].nunique()/unique_groups:.1%}), test={test[GROUP].nunique():,} ({test[GROUP].nunique()/unique_groups:.1%})')
    print(f'Driver overlap: {len(overlap)}')
    print(f'TV distance train/test: {best["tv_tr"]:.4f}/{best["tv_te"]:.4f}')
    print(f'Missing classes train/test: {best["missing_train"]}/{best["missing_test"]}')
    print(f'Saved: {out_csv.resolve()}')
    print(f'Saved: {(OUTDIR / "split_audit.csv").resolve()}')
    print(f'Saved: {(OUTDIR / "label_distribution_comparison.csv").resolve()}')
    print('Do not train on this split until the audit is inspected.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
