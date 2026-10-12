import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="day4_feature_ablation")
    args = parser.parse_args()

    folder = Path(args.input_dir)
    files = sorted(folder.glob("*_predictions.csv"))

    if not files:
        raise FileNotFoundError(f"No prediction CSV files found in {folder}")

    driver_results = []
    class_results = []

    for file in files:
        df = pd.read_csv(file)
        required = {"userid", "actual", "predicted"}
        if not required.issubset(df.columns):
            print(f"Skipping {file.name}: required columns missing")
            continue

        model = str(df["model"].iloc[0])
        feature_group = str(df["feature_group"].iloc[0])
        labels = sorted(df["actual"].dropna().astype(str).unique())

        df["actual"] = df["actual"].astype(str)
        df["predicted"] = df["predicted"].astype(str)

        for driver, group in df.groupby("userid"):
            driver_results.append({
                "model": model,
                "feature_group": feature_group,
                "userid": driver,
                "rows": len(group),
                "accuracy": accuracy_score(group["actual"], group["predicted"]),
                "balanced_accuracy": balanced_accuracy_score(
                    group["actual"], group["predicted"]
                ),
                "macro_f1": f1_score(
                    group["actual"], group["predicted"],
                    labels=labels, average="macro", zero_division=0
                ),
                "classes_present": group["actual"].nunique(),
            })

        report = []
        from sklearn.metrics import classification_report
        report = classification_report(
            df["actual"], df["predicted"],
            labels=labels, output_dict=True, zero_division=0
        )

        for label in labels:
            class_results.append({
                "model": model,
                "feature_group": feature_group,
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": report[label]["support"],
            })

    per_driver = pd.DataFrame(driver_results)
    per_class = pd.DataFrame(class_results)

    per_driver.to_csv(folder / "per_driver_metrics.csv", index=False)
    per_class.to_csv(folder / "per_class_metrics_recomputed.csv", index=False)

    summary = (
        per_driver.groupby(["model", "feature_group"])
        .agg(
            drivers=("userid", "nunique"),
            mean_driver_accuracy=("accuracy", "mean"),
            median_driver_accuracy=("accuracy", "median"),
            mean_driver_macro_f1=("macro_f1", "mean"),
            median_driver_macro_f1=("macro_f1", "median"),
            mean_driver_balanced_accuracy=("balanced_accuracy", "mean"),
            median_rows_per_driver=("rows", "median"),
        )
        .reset_index()
    )
    summary.to_csv(folder / "driver_level_summary.csv", index=False)

    print("\n=== DRIVER-LEVEL SUMMARY ===")
    print(summary.round(4).to_string(index=False))
    print("\nSaved:")
    print(folder / "per_driver_metrics.csv")
    print(folder / "per_class_metrics_recomputed.csv")
    print(folder / "driver_level_summary.csv")

if __name__ == "__main__":
    main()
