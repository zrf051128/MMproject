import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_utils import RESULTS_DIR


METHOD_ORDER = [
    "opencv_telea",
    "opencv_ns",
    "tv",
    "wavelet",
    "kmeans_patch",
    "ours",
]

METHOD_LABELS = {
    "opencv_telea": "OpenCV Telea",
    "opencv_ns": "OpenCV NS",
    "tv": "TV",
    "wavelet": "Wavelet",
    "kmeans_patch": "K-means Patch",
    "ours": "Ours",
}

MASK_ORDER = ["random10", "random30", "random50", "irregular"]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate summary tables and figures from formal experiment CSV files."
    )
    parser.add_argument(
        "--main_csv",
        type=str,
        default=str(RESULTS_DIR / "tables" / "main_results.csv"),
    )
    parser.add_argument(
        "--objective_csv",
        type=str,
        default=str(RESULTS_DIR / "tables" / "objective_ablation.csv"),
    )
    parser.add_argument(
        "--lambda_csv",
        type=str,
        default=str(RESULTS_DIR / "tables" / "lambda_ae_sensitivity.csv"),
    )
    return parser


def method_sort_key(method):
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER)


def read_valid_results(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"]
    df = df[df["method"].isin(METHOD_ORDER)]
    return df


def write_method_average_table(df, out_path):
    grouped = (
        df.groupby("method", as_index=False)
        .agg(
            psnr=("psnr", "mean"),
            ssim=("ssim", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            runtime=("runtime", "mean"),
        )
    )
    grouped["method_order"] = grouped["method"].map(method_sort_key)
    grouped = grouped.sort_values("method_order").drop(columns=["method_order"])
    grouped["method_label"] = grouped["method"].map(lambda x: METHOD_LABELS.get(x, x))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    return grouped


def write_psnr_by_mask_table(df, out_path):
    table = df.pivot_table(
        index="method",
        columns="mask_type",
        values="psnr",
        aggfunc="mean",
    )
    ordered_methods = [m for m in METHOD_ORDER if m in table.index]
    ordered_masks = [m for m in MASK_ORDER if m in table.columns]
    table = table.loc[ordered_methods, ordered_masks]
    table.insert(0, "method_label", [METHOD_LABELS.get(m, m) for m in table.index])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path)
    return table


def plot_method_psnr_bar(avg_df, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = avg_df["method_label"].tolist()
    values = avg_df["psnr"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylabel("Mean PSNR")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_psnr_by_mask(mask_table, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = mask_table.drop(columns=["method_label"]).copy()
    x = np.arange(len(plot_df.columns))
    width = 0.12

    plt.figure(figsize=(9, 4.5))
    for idx, method in enumerate(plot_df.index):
        offset = (idx - (len(plot_df.index) - 1) / 2) * width
        plt.bar(
            x + offset,
            plot_df.loc[method].to_numpy(dtype=float),
            width=width,
            label=METHOD_LABELS.get(method, method),
        )
    plt.xticks(x, plot_df.columns)
    plt.ylabel("Mean PSNR")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_lambda_sensitivity(lambda_csv, out_path):
    lambda_csv = Path(lambda_csv)
    if not lambda_csv.exists():
        print(f"Skipping lambda plot; missing {lambda_csv}")
        return

    df = pd.read_csv(lambda_csv)
    if "lambda_ae" not in df.columns:
        print(f"Skipping lambda plot; no lambda_ae column in {lambda_csv}")
        return
    df = df.sort_values("lambda_ae")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.semilogx(df["lambda_ae"], df["psnr"], marker="o")
    plt.xlabel("lambda_AE")
    plt.ylabel("PSNR")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_objective_table(objective_csv, out_path):
    objective_csv = Path(objective_csv)
    if not objective_csv.exists():
        print(f"Skipping objective table; missing {objective_csv}")
        return None
    df = pd.read_csv(objective_csv)
    keep = ["image", "mask_type", "method", "psnr", "ssim", "rmse", "mae", "runtime"]
    keep = [c for c in keep if c in df.columns]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(out_path, index=False)
    return df[keep]


def main():
    args = build_arg_parser().parse_args()
    tables_dir = RESULTS_DIR / "tables"
    figures_dir = RESULTS_DIR / "figures"

    df = read_valid_results(args.main_csv)
    avg_df = write_method_average_table(df, tables_dir / "table1_method_averages.csv")
    mask_table = write_psnr_by_mask_table(df, tables_dir / "table2_psnr_by_mask.csv")

    plot_method_psnr_bar(avg_df, figures_dir / "table1_method_psnr.png")
    plot_psnr_by_mask(mask_table, figures_dir / "table2_psnr_by_mask.png")
    plot_lambda_sensitivity(args.lambda_csv, figures_dir / "lambda_ae_sensitivity.png")
    write_objective_table(args.objective_csv, tables_dir / "objective_ablation_for_paper.csv")

    print(f"Wrote: {tables_dir / 'table1_method_averages.csv'}")
    print(f"Wrote: {tables_dir / 'table2_psnr_by_mask.csv'}")
    print(f"Wrote: {figures_dir / 'table1_method_psnr.png'}")
    print(f"Wrote: {figures_dir / 'table2_psnr_by_mask.png'}")


if __name__ == "__main__":
    main()
