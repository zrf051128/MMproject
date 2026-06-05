import argparse

import numpy as np
import pandas as pd

from experiment_utils import (
    RESULTS_DIR,
    append_csv_rows,
    compute_all_metrics,
    default_method_settings,
    ensure_result_dirs,
    get_autoencoder_for_case,
    load_selected_images,
    make_mask,
    metric_row,
    plot_loss_curve,
    plot_psnr_curve,
    run_restoration_method,
    save_comparison_figure,
    save_image,
    save_log,
)
from src.baselines import tv_inpainting
from src.mask_generator import apply_mask


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run formal objective ablation and lambda_AE sensitivity experiments."
    )
    parser.add_argument("--image", type=str, default="cameraman")
    parser.add_argument("--mask", type=str, default="random30")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)

    parser.add_argument(
        "--ae_source",
        type=str,
        default="load",
        choices=["load", "train", "auto"],
        help="load=image-level weights, train=mask-specific AE, auto=mask-specific then image-level then train.",
    )
    parser.add_argument("--ae_epochs", type=int, default=50)
    parser.add_argument("--ae_batch_size", type=int, default=128)
    parser.add_argument("--ae_lr", type=float, default=1e-3)

    parser.add_argument("--lam_ae", type=float, default=1e-3)
    parser.add_argument("--lam_tv", type=float, default=1e-3)
    parser.add_argument(
        "--lambda_sweep",
        type=str,
        default="1e-4,1e-3,1e-2,5e-2,1e-1",
        help="Comma-separated lambda_AE values.",
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tv_steps", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def settings_from_args(args):
    settings = default_method_settings()
    settings.update(
        {
            "lam_ae": args.lam_ae,
            "lam_tv": args.lam_tv,
            "ours_steps": args.steps,
            "ours_lr": args.lr,
            "tv_steps": args.tv_steps,
            "tv_lr": args.lr,
            "log_interval": args.log_interval,
        }
    )
    return settings


def save_ablation_outputs(prefix, x_hat, log, row, dirs, comparison_images, comparison_titles):
    image_path = dirs["restored"] / f"{prefix}.png"
    log_path = dirs["logs"] / f"{prefix}_log.npy"
    save_image(x_hat, image_path)
    save_log(log, log_path)
    comparison_images.append(x_hat)
    comparison_titles.append(row["method"])


def main():
    args = build_arg_parser().parse_args()
    dirs = ensure_result_dirs()
    settings = settings_from_args(args)

    images = load_selected_images([args.image], image_size=args.image_size)
    x_gt = images[args.image]
    mask = make_mask(args.mask, x_gt.shape, seed=args.seed)
    y = apply_mask(x_gt, mask)

    prefix_base = f"{args.image}_{args.mask}"
    save_image(x_gt, dirs["restored"] / f"{prefix_base}_original.png")
    save_image(mask, dirs["restored"] / f"{prefix_base}_mask.png")
    save_image(y, dirs["restored"] / f"{prefix_base}_corrupted.png")

    ae, ae_weight_used = get_autoencoder_for_case(
        image_name=args.image,
        mask_type=args.mask,
        clean_image=x_gt,
        corrupted_image=y,
        mask=mask,
        ae_source=args.ae_source,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        stride=args.stride,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_lr=args.ae_lr,
        verbose=args.verbose,
    )

    objective_rows = []
    comparison_images = [x_gt, mask, y]
    comparison_titles = ["Original", "Mask", "Corrupted"]

    corrupted_metrics = compute_all_metrics(y, x_gt)
    objective_rows.append(
        metric_row(args.image, args.mask, "corrupted", corrupted_metrics, 0.0)
    )

    print("Running TV only...")
    import time

    start = time.time()
    x_tv, tv_log = tv_inpainting(
        y,
        mask,
        lam_tv=args.lam_tv,
        lr=args.lr,
        steps=args.tv_steps,
        init="telea",
        verbose=args.verbose,
    )
    runtime = time.time() - start
    tv_metrics = compute_all_metrics(x_tv, x_gt)
    tv_row = metric_row(args.image, args.mask, "tv_only", tv_metrics, runtime)
    objective_rows.append(tv_row)
    save_ablation_outputs(
        f"{prefix_base}_ablation_tv_only",
        x_tv,
        tv_log,
        tv_row,
        dirs,
        comparison_images,
        comparison_titles,
    )

    print("Running AE only...")
    ae_only_settings = dict(settings)
    ae_only_settings["lam_tv"] = 0.0
    ae_only_settings["lam_ae"] = args.lam_ae
    x_ae, ae_log, ae_runtime, ae_metrics = run_restoration_method(
        "ours",
        y,
        mask,
        x_gt,
        ae_only_settings,
        ae=ae,
        patch_size=args.patch_size,
        stride=args.stride,
        verbose=args.verbose,
    )
    ae_row = metric_row(
        args.image,
        args.mask,
        "ae_only",
        ae_metrics,
        ae_runtime,
        extra={"ae_weight": ae_weight_used},
    )
    objective_rows.append(ae_row)
    save_ablation_outputs(
        f"{prefix_base}_ablation_ae_only",
        x_ae,
        ae_log,
        ae_row,
        dirs,
        comparison_images,
        comparison_titles,
    )

    print("Running AE + TV...")
    ae_tv_settings = dict(settings)
    ae_tv_settings["lam_tv"] = args.lam_tv
    ae_tv_settings["lam_ae"] = args.lam_ae
    x_ae_tv, ae_tv_log, ae_tv_runtime, ae_tv_metrics = run_restoration_method(
        "ours",
        y,
        mask,
        x_gt,
        ae_tv_settings,
        ae=ae,
        patch_size=args.patch_size,
        stride=args.stride,
        verbose=args.verbose,
    )
    ae_tv_row = metric_row(
        args.image,
        args.mask,
        "ae_tv",
        ae_tv_metrics,
        ae_tv_runtime,
        extra={"ae_weight": ae_weight_used},
    )
    objective_rows.append(ae_tv_row)
    save_ablation_outputs(
        f"{prefix_base}_ablation_ae_tv",
        x_ae_tv,
        ae_tv_log,
        ae_tv_row,
        dirs,
        comparison_images,
        comparison_titles,
    )

    objective_csv = dirs["tables"] / "objective_ablation.csv"
    append_csv_rows(objective_rows, objective_csv)
    save_comparison_figure(
        comparison_images,
        comparison_titles,
        RESULTS_DIR / "figures" / f"{prefix_base}_objective_ablation.png",
        max_cols=3,
    )
    plot_loss_curve(
        ae_tv_log,
        RESULTS_DIR / "figures" / f"{prefix_base}_convergence_loss.png",
        title=f"{prefix_base} AE+TV Loss",
    )
    plot_psnr_curve(
        ae_tv_log,
        RESULTS_DIR / "figures" / f"{prefix_base}_convergence_psnr.png",
        title=f"{prefix_base} AE+TV PSNR",
    )

    print("Running lambda_AE sensitivity...")
    sweep_rows = []
    sweep_images = [x_gt, y]
    sweep_titles = ["Original", "Corrupted"]

    for lam_ae in parse_float_list(args.lambda_sweep):
        sweep_settings = dict(settings)
        sweep_settings["lam_ae"] = lam_ae
        sweep_settings["lam_tv"] = args.lam_tv

        x_sweep, sweep_log, sweep_runtime, sweep_metrics = run_restoration_method(
            "ours",
            y,
            mask,
            x_gt,
            sweep_settings,
            ae=ae,
            patch_size=args.patch_size,
            stride=args.stride,
            verbose=args.verbose,
        )

        lam_label = f"{lam_ae:.0e}".replace("+", "")
        row = metric_row(
            args.image,
            args.mask,
            f"lambda_ae_{lam_label}",
            sweep_metrics,
            sweep_runtime,
            extra={"lambda_ae": lam_ae, "ae_weight": ae_weight_used},
        )
        sweep_rows.append(row)

        save_image(x_sweep, dirs["restored"] / f"{prefix_base}_lambda_ae_{lam_label}.png")
        save_log(sweep_log, dirs["logs"] / f"{prefix_base}_lambda_ae_{lam_label}_log.npy")
        sweep_images.append(x_sweep)
        sweep_titles.append(f"lam_AE={lam_ae:g}")

        print(
            f"  lambda_AE={lam_ae:g}: PSNR={row['psnr']:.4f} "
            f"SSIM={row['ssim']:.4f} runtime={row['runtime']:.2f}s"
        )

    sweep_csv = dirs["tables"] / "lambda_ae_sensitivity.csv"
    append_csv_rows(sweep_rows, sweep_csv)
    save_comparison_figure(
        sweep_images,
        sweep_titles,
        RESULTS_DIR / "figures" / f"{prefix_base}_lambda_ae_sensitivity.png",
        max_cols=4,
    )

    print(f"Saved objective ablation: {objective_csv}")
    print(f"Saved lambda sensitivity: {sweep_csv}")
    print(pd.DataFrame(objective_rows))
    print(pd.DataFrame(sweep_rows))


if __name__ == "__main__":
    main()
