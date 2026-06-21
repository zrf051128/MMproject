# Image AE Inpainting Experiments

This repository contains code for an image inpainting experiment based on optimization methods and an autoencoder prior. The project compares several restoration methods on grayscale images with missing pixels or irregular masks.

The main comparison methods are:

- DCT-only
- Wavelet-only
- TV
- AE-only
- TV+AE

The default experiment images are `cameraman`, `barbara`, `house`, and `peppers`.

## Project Structure

```text
MMproject/
|-- data/
|   `-- images/                  # Optional local classic images
|-- image_ae_inpainting/
|   |-- scripts/
|   |   |-- run_main_experiments.py
|   |   |-- run_ablation_experiments.py
|   |   |-- plot_results.py
|   |   |-- experiment_utils.py
|   |   `-- check_*.py
|   `-- src/
|       |-- autoencoder.py
|       |-- baselines.py
|       |-- data_loader.py
|       |-- mask_generator.py
|       |-- metrics.py
|       |-- optimization.py
|       |-- patch_utils.py
|       `-- train_ae.py
|-- results/
|   |-- ae_weights/
|   |-- figures/
|   |-- logs/
|   |-- restored_images/
|   `-- tables/
`-- README.md
```

## Environment

Use Python 3.10 or newer. From the project root, install the required packages:

```powershell
pip install numpy pandas matplotlib pillow scikit-image torch
```

If you need a CUDA-enabled PyTorch build, install PyTorch according to your local CUDA version.

## Data And Weights

Run all commands from the project root:

```powershell
cd D:\pythonProject\MMproject
```

The current code expects paths relative to this root directory.

Local image files should be placed in:

```text
data/images/
```

For example:

```text
data/images/barbara.png
data/images/house.png
data/images/peppers.png
```

`cameraman` is loaded from `scikit-image`, so it does not need a local image file.

The default trained autoencoder weights are expected in:

```text
results/ae_weights/
```

Current expected combined weight names:

```text
barbara_p8_d16_combined.pt
cameraman_p8_d16_combined.pt
house_p8_d16_combined.pt
peppers_p8_d16_combined.pt
```

## Quick Test

Run a small test first to confirm the environment and paths are correct:

```powershell
python image_ae_inpainting\scripts\run_main_experiments.py --images cameraman --masks random10 --methods dct,tv --max_cases 1 --steps 50
```

This should create result files under `results/`.

## Main Experiments

Run the full main experiment with the default images, masks, and methods:

```powershell
python image_ae_inpainting\scripts\run_main_experiments.py --ae_source load
```

Use `--ae_source load` when the trained weights already exist in `results/ae_weights/`.

If weights are missing and you want the script to train when needed:

```powershell
python image_ae_inpainting\scripts\run_main_experiments.py --ae_source auto
```

Useful options:

```powershell
python image_ae_inpainting\scripts\run_main_experiments.py ^
  --images cameraman,barbara,house,peppers ^
  --masks random10,random30,random50,irregular ^
  --methods dct,wavelet,tv,ae_only,tv_ae ^
  --patch_size 8 ^
  --latent_dim 16 ^
  --steps 500 ^
  --lr 0.03 ^
  --ae_source load
```

## Ablation Experiments

Run the ablation experiment for one image and one mask:

```powershell
python image_ae_inpainting\scripts\run_ablation_experiments.py --image cameraman --mask random30 --ae_source load
```

You can change the lambda sweep:

```powershell
python image_ae_inpainting\scripts\run_ablation_experiments.py ^
  --image cameraman ^
  --mask random30 ^
  --lambda_sweep 1e-4,3e-4,1e-3,3e-3,1e-2 ^
  --ae_source load
```

## Plot Results

After running experiments, generate summary tables and figures:

```powershell
python image_ae_inpainting\scripts\plot_results.py
```

## Outputs

Main experiment outputs:

```text
results/tables/main_results.csv
results/restored_images/
results/logs/
```

Ablation outputs:

```text
results/tables/objective_ablation.csv
results/tables/lambda_ae_sensitivity.csv
```

Plotting outputs:

```text
results/tables/table1_method_averages.csv
results/tables/table2_psnr_by_mask.csv
results/tables/objective_ablation_for_paper.csv
results/figures/table1_method_psnr.png
results/figures/table2_psnr_by_mask.png
results/figures/lambda_ae_sensitivity.png
```

## Evaluation Metrics

The experiments report several image restoration metrics:

- `psnr`: peak signal-to-noise ratio on the full image.
- `ssim`: structural similarity on the full image.
- `rmse`: root mean squared error on the full image.
- `mae`: mean absolute error on the full image.
- `psnr_missing`: PSNR on the missing region only.
- `rmse_missing`: RMSE on the missing region only.
- `mae_missing`: MAE on the missing region only.

For inpainting, the missing-region metrics are especially important because they directly measure the reconstructed area.


## Authors

This project is for a Mathematical Modeling course project by Ruofan Zhang and Hanlin Geng.
