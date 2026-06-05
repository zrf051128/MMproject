import argparse
import traceback

import numpy as np

from experiment_utils import (
    DEFAULT_IMAGES,
    DEFAULT_MASKS,
    DEFAULT_METHODS,
    RESULTS_DIR,
    append_csv_rows,
    compute_all_metrics,
    default_method_settings,
    ensure_result_dirs,
    get_autoencoder_for_case,
    load_selected_images,
    make_mask,
    metric_row,
    parse_name_list,
    plot_loss_curve,
    plot_psnr_curve,
    run_restoration_method,
    save_comparison_figure,
    save_image,
    save_log,
)
from src.mask_generator import apply_mask


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run formal image AE inpainting main experiments."
    )
    parser.add_argument(
        "--images",
        type=str,
        default=",".join(DEFAULT_IMAGES),
        help="Comma-separated image names, or 'all'.",
    )
    parser.add_argument(
        "--masks",
        type=str,
        default=",".join(DEFAULT_MASKS),
        help="Comma-separated mask names, e.g. random10,random30,random50,irregular.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods: opencv_telea,opencv_ns,tv,wavelet,kmeans_patch,ours.",
    )
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
    parser.add_argument("--ours_steps", type=int, default=800)
    parser.add_argument("--ours_lr", type=float, default=1e-2)
    parser.add_argument("--tv_steps", type=int, default=500)
    parser.add_argument("--tv_lr", type=float, default=1e-2)
    parser.add_argument("--wavelet_steps", type=int, default=120)
    parser.add_argument("--lam_wavelet", type=float, default=0.01)
    parser.add_argument("--kmeans_clusters", type=int, default=64)
    parser.add_argument("--kmeans_iters", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=20)

    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="Optional limit on image-mask cases for quick smoke tests. 0 means no limit.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a method if its restored image already exists.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when a method fails.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def settings_from_args(args):
    settings = default_method_settings()
    settings.update(
        {
            "lam_ae": args.lam_ae,
            "lam_tv": args.lam_tv,
            "ours_steps": args.ours_steps,
            "ours_lr": args.ours_lr,
            "tv_steps": args.tv_steps,
            "tv_lr": args.tv_lr,
            "wavelet_steps": args.wavelet_steps,
            "lam_wavelet": args.lam_wavelet,
            "kmeans_clusters": args.kmeans_clusters,
            "kmeans_iters": args.kmeans_iters,
            "log_interval": args.log_interval,
        }
    )
    return settings


def main():
    args = build_arg_parser().parse_args()

    image_names = parse_name_list(args.images, DEFAULT_IMAGES)
    mask_types = parse_name_list(args.masks, DEFAULT_MASKS)
    methods = parse_name_list(args.methods, DEFAULT_METHODS)
    dirs = ensure_result_dirs()
    settings = settings_from_args(args)

    images = load_selected_images(image_names, image_size=args.image_size)
    rows = []
    case_count = 0

    for image_name, x_gt in images.items():
        for mask_type in mask_types:
            if args.max_cases and case_count >= args.max_cases:
                break
            case_count += 1

            print("=" * 70)
            print(f"Case {case_count}: image={image_name}, mask={mask_type}")
            print("=" * 70)

            mask = make_mask(mask_type, x_gt.shape, seed=args.seed)
            y = apply_mask(x_gt, mask)

            save_image(x_gt, dirs["restored"] / f"{image_name}_{mask_type}_original.png")
            save_image(mask, dirs["restored"] / f"{image_name}_{mask_type}_mask.png")
            save_image(y, dirs["restored"] / f"{image_name}_{mask_type}_corrupted.png")

            corrupted_metrics = compute_all_metrics(y, x_gt)
            rows.append(
                metric_row(
                    image_name,
                    mask_type,
                    "corrupted",
                    corrupted_metrics,
                    0.0,
                    extra={"status": "ok", "ae_weight": ""},
                )
            )

            comparison_images = [x_gt, mask, y]
            comparison_titles = ["Original", "Mask", "Corrupted"]

            ae = None
            ae_weight_used = ""
            if "ours" in [m.lower() for m in methods]:
                try:
                    ae, ae_weight_used = get_autoencoder_for_case(
                        image_name=image_name,
                        mask_type=mask_type,
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
                except Exception as exc:
                    if args.stop_on_error:
                        raise
                    print(f"[ERROR] Failed to prepare AE for {image_name}/{mask_type}: {exc}")
                    ae = None
                    ae_weight_used = ""

            for method in methods:
                method = method.lower()
                output_path = dirs["restored"] / f"{image_name}_{mask_type}_{method}.png"
                log_path = dirs["logs"] / f"{image_name}_{mask_type}_{method}_log.npy"

                if args.skip_existing and output_path.exists():
                    print(f"[skip] {output_path}")
                    continue

                print(f"Running method: {method}")
                try:
                    if method == "ours" and ae is None:
                        raise RuntimeError("AE was not prepared; cannot run ours.")

                    x_hat, log, runtime, metrics = run_restoration_method(
                        method,
                        y,
                        mask,
                        x_gt,
                        settings,
                        ae=ae,
                        patch_size=args.patch_size,
                        stride=args.stride,
                        verbose=args.verbose,
                    )

                    save_image(x_hat, output_path)
                    save_log(log, log_path)
                    comparison_images.append(x_hat)
                    comparison_titles.append(method)

                    if method == "ours" and log:
                        plot_loss_curve(
                            log,
                            RESULTS_DIR / "figures" / f"{image_name}_{mask_type}_ours_loss.png",
                            title=f"{image_name} {mask_type} Ours Loss",
                        )
                        plot_psnr_curve(
                            log,
                            RESULTS_DIR / "figures" / f"{image_name}_{mask_type}_ours_psnr.png",
                            title=f"{image_name} {mask_type} Ours PSNR",
                        )

                    row = metric_row(
                        image_name,
                        mask_type,
                        method,
                        metrics,
                        runtime,
                        extra={"status": "ok", "ae_weight": ae_weight_used if method == "ours" else ""},
                    )
                    rows.append(row)
                    print(
                        f"  PSNR={row['psnr']:.4f} SSIM={row['ssim']:.4f} "
                        f"RMSE={row['rmse']:.4f} runtime={row['runtime']:.2f}s"
                    )

                except Exception as exc:
                    if args.stop_on_error:
                        raise
                    print(f"[ERROR] {image_name}/{mask_type}/{method}: {exc}")
                    traceback.print_exc()
                    rows.append(
                        {
                            "image": image_name,
                            "mask_type": mask_type,
                            "method": method,
                            "psnr": np.nan,
                            "ssim": np.nan,
                            "rmse": np.nan,
                            "mae": np.nan,
                            "runtime": np.nan,
                            "status": "error",
                            "ae_weight": ae_weight_used if method == "ours" else "",
                            "error": str(exc),
                        }
                    )

            save_comparison_figure(
                comparison_images,
                comparison_titles,
                RESULTS_DIR / "figures" / f"{image_name}_{mask_type}_comparison.png",
                max_cols=4,
            )

        if args.max_cases and case_count >= args.max_cases:
            break

    csv_path = dirs["tables"] / "main_results.csv"
    append_csv_rows(rows, csv_path)
    print(f"Saved main results to: {csv_path}")
    print(f"Restored images: {dirs['restored']}")
    print(f"Logs: {dirs['logs']}")
    print(f"Figures: {dirs['figures']}")


if __name__ == "__main__":
    main()
