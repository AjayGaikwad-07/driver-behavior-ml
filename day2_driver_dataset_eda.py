"""Day 2: memory-safe EDA starter for large driver-behavior CSV files.
Install: pip install pandas numpy matplotlib
Quick scan: python day2_driver_dataset_eda.py "path/to/DriverSVT.csv"
Full scan:  python day2_driver_dataset_eda.py "path/to/DriverSVT.csv" --full
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--outdir", default="day2_eda_outputs")
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--max-chunks", type=int, default=5)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sample-rows", type=int, default=25_000)
    args = ap.parse_args()
    path = Path(args.csv_path).expanduser()
    out = Path(args.outdir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    missing, labels, users = Counter(), Counter(), Counter()
    sums, sumsq, counts = Counter(), Counter(), Counter()
    sample_parts, sample_kept = [], 0
    rows = chunks = 0
    columns = None
    label_col = user_col = None

    try:
        for chunk in pd.read_csv(path, chunksize=args.chunksize, low_memory=False):
            chunks += 1
            rows += len(chunk)
            if columns is None:
                columns = [str(c) for c in chunk.columns]
                lower = {c.lower(): c for c in columns}
                label_col = lower.get("dangerousstate")
                user_col = lower.get("userid")
            for col in chunk.columns:
                missing[str(col)] += int(chunk[col].isna().sum())
            if label_col:
                labels.update(chunk[label_col].fillna("<MISSING>").astype(str).value_counts().to_dict())
            if user_col:
                users.update(chunk[user_col].fillna("<MISSING>").astype(str).value_counts().to_dict())
            for col in chunk.select_dtypes(include=[np.number]).columns:
                s = chunk[col].dropna().astype("float64")
                if len(s):
                    sums[str(col)] += float(s.sum())
                    sumsq[str(col)] += float((s * s).sum())
                    counts[str(col)] += int(s.count())
            if sample_kept < args.sample_rows:
                part = chunk.head(args.sample_rows - sample_kept).copy()
                sample_parts.append(part)
                sample_kept += len(part)
            if not args.full and chunks >= args.max_chunks:
                break
            if chunks % 10 == 0:
                print(f"Scanned {rows:,} rows...")
    except Exception as exc:
        print(f"ERROR reading CSV: {exc}", file=sys.stderr)
        return 1

    if not columns:
        print("ERROR: empty CSV", file=sys.stderr)
        return 1

    scan = "FULL FILE SCAN" if args.full else f"QUICK SCAN (first {rows:,} rows only)"
    overview = [
        "DAY 2 — DRIVER DATASET EDA", f"File: {path.resolve()}",
        f"File size: {path.stat().st_size:,} bytes", f"Scan: {scan}",
        f"Rows scanned: {rows:,}", f"Chunks scanned: {chunks:,}",
        f"Columns ({len(columns)}): {', '.join(columns)}",
        f"Detected target field: {label_col or 'not found'}",
        f"Detected driver field: {user_col or 'not found'}",
        "Quick-scan counts and missingness are for the scanned rows only."
    ]
    (out / "overview.txt").write_text("\n".join(overview), encoding="utf-8")
    miss_df = pd.DataFrame({
        "column": columns,
        "missing_count_scanned_rows": [missing.get(c, 0) for c in columns]
    })
    miss_df["missing_percent_scanned_rows"] = 100 * miss_df["missing_count_scanned_rows"] / max(rows, 1)
    miss_df.sort_values("missing_percent_scanned_rows", ascending=False).to_csv(out / "missing_values.csv", index=False)
    if labels:
        pd.DataFrame(labels.most_common(), columns=["dangerousstate_value", "count_scanned_rows"]).to_csv(out / "dangerousstate_counts.csv", index=False)
    if users:
        pd.DataFrame(users.most_common(), columns=["userid", "count_scanned_rows"]).to_csv(out / "driver_counts.csv", index=False)
    stats = []
    for col, n in counts.items():
        mean = sums[col] / n
        var = max(sumsq[col] / n - mean * mean, 0)
        stats.append({"column": col, "count": n, "mean": mean, "std_approx": var ** 0.5})
    if stats:
        pd.DataFrame(stats).sort_values("column").to_csv(out / "numeric_summary.csv", index=False)

    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    if not sample.empty:
        sample.head(2000).to_csv(out / "sample_first_rows.csv", index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if label_col and label_col in sample:
                vc = sample[label_col].fillna("<MISSING>").astype(str).value_counts().head(30)
                fig, ax = plt.subplots(figsize=(10, 5))
                vc.plot(kind="bar", ax=ax)
                ax.set_title("Driver-state label counts — sample only")
                ax.set_xlabel("Label")
                ax.set_ylabel("Rows in sample")
                fig.tight_layout()
                fig.savefig(out / "sample_label_distribution.png", dpi=150)
                plt.close(fig)
            signal_cols = [c for c in ["speed", "acc_X", "acc_Y", "acc_Z", "perclos", "face_mouth"] if c in sample]
            if signal_cols:
                n = min(len(sample), 2000)
                fig, axes = plt.subplots(len(signal_cols), 1, figsize=(11, 2.4 * len(signal_cols)))
                if len(signal_cols) == 1:
                    axes = [axes]
                for ax, col in zip(axes, signal_cols):
                    vals = pd.to_numeric(sample[col].head(n), errors="coerce")
                    ax.plot(np.arange(len(vals)), vals, linewidth=0.8)
                    ax.set_title(f"{col} — sample rows")
                    ax.set_xlabel("Row index (not normalized time)")
                fig.tight_layout()
                fig.savefig(out / "sample_sensor_signals.png", dpi=150)
                plt.close(fig)
        except ImportError:
            print("matplotlib not installed; CSV reports were still saved.")

    print(f"Rows scanned: {rows:,} ({'full file' if args.full else 'quick sample only'})")
    print(f"Columns: {len(columns)}")
    print(f"Results saved to: {out.resolve()}")
    print("Read overview.txt first.")
    if not args.full:
        print("Run with --full later to scan all rows in chunks.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
