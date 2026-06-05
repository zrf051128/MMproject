import numpy as np


def _to_float01(x):
    x = np.asarray(x).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def torch_tv_loss(x, eps=1e-6):
    """
    Isotropic TV loss.

    x shape: [1, 1, H, W]
    """
    import torch

    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]

    dx_crop = dx[:, :, :-1, :]
    dy_crop = dy[:, :, :, :-1]
    return torch.sqrt(dx_crop ** 2 + dy_crop ** 2 + eps).mean()


def extract_patches_torch(x, patch_size=8, stride=4):
    """
    Extract overlapping patches using torch unfold.

    Parameters
    ----------
    x : torch.Tensor
        Shape [1, 1, H, W].

    Returns
    -------
    patches_img : torch.Tensor
        Shape [N, 1, patch_size, patch_size].
    patches_flat : torch.Tensor
        Shape [N, patch_size * patch_size].
    """
    import torch.nn.functional as F

    patches = F.unfold(
        x,
        kernel_size=patch_size,
        stride=stride
    )

    # patches: [1, patch_size * patch_size, N]
    patches = patches.squeeze(0).transpose(0, 1)

    # patches_flat: [N, patch_size * patch_size]
    patches_flat = patches

    # patches_img: [N, 1, patch_size, patch_size]
    patches_img = patches_flat.view(-1, 1, patch_size, patch_size)

    return patches_img, patches_flat


def ae_patch_loss(x, ae, patch_size=8, stride=4, ae_input_format="image"):
    """
    Compute AE patch prior:

        R_AE(x) = sum_i || P_i x - A_theta(P_i x) ||^2

    ae_input_format:
        "image": AE input shape [N, 1, patch_size, patch_size]
        "flat":  AE input shape [N, patch_size * patch_size]
    """
    patches_img, patches_flat = extract_patches_torch(
        x,
        patch_size=patch_size,
        stride=stride
    )

    if ae_input_format == "image":
        ae_input = patches_img
        target = patches_img
    elif ae_input_format == "flat":
        ae_input = patches_flat
        target = patches_flat
    else:
        raise ValueError("ae_input_format must be 'image' or 'flat'.")

    recon = ae(ae_input)

    # Make output shape compatible
    if recon.shape != target.shape:
        if recon.ndim == 2:
            recon = recon.view_as(patches_flat)
            target = patches_flat
        elif recon.ndim == 4:
            recon = recon.view_as(patches_img)
            target = patches_img
        else:
            raise ValueError(f"Unexpected AE output shape: {recon.shape}")

    loss = ((recon - target) ** 2).mean()
    return loss


def optimize_image_with_ae_tv(
    y,
    mask,
    ae,
    patch_size=8,
    stride=4,
    lam_ae=0.01,
    lam_tv=0.01,
    lr=0.03,
    steps=500,
    init="mean",
    ae_input_format="image",
    fix_observed=True,
    x_gt=None,
    log_interval=20,
    verbose=True,
):
    """
    AE-only or TV+AE optimization, depending on lambda values.

    Optimization model:

        min_x  0.5 || M ⊙ x - y ||_2^2
             + lambda_AE R_AE(x)
             + lambda_TV TV(x)

    Parameters
    ----------
    y : np.ndarray
        Corrupted image, shape [H, W], range [0, 1].
    mask : np.ndarray
        Binary mask, shape [H, W].
        mask = 1 means observed.
        mask = 0 means missing.
    ae : torch.nn.Module
        Trained patch autoencoder.
    patch_size : int
        Patch size.
    stride : int
        Patch stride.
    lam_ae : float
        Weight for AE patch prior.
    lam_tv : float
        Weight for TV prior.
    lr : float
        Adam learning rate.
    steps : int
        Optimization steps.
    init : str
        Initialization method. Supports "mean" and "zero".
    ae_input_format : str
        "image" or "flat".
    fix_observed : bool
        If True, observed pixels are fixed exactly to y after each update.
    x_gt : np.ndarray or None
        Ground truth image. Only used for logging PSNR/SSIM.
    log_interval : int
        Record PSNR/SSIM every log_interval steps.
    verbose : bool
        Print progress.

    Returns
    -------
    x_hat : np.ndarray
        Restored image.
    log : dict
        Optimization log.
    """
    import torch

    y = _to_float01(y)
    mask = np.asarray(mask).astype(np.float32)

    H, W = y.shape

    if init == "zero":
        x0 = y.copy()
    elif init == "mean":
        x0 = y.copy()
        observed = y[mask == 1]
        mean_val = observed.mean() if observed.size > 0 else 0.0
        x0[mask == 0] = mean_val
        x0 = _to_float01(x0)
    else:
        raise ValueError("init must be 'mean' or 'zero'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_t = torch.tensor(y, dtype=torch.float32, device=device).view(1, 1, H, W)
    m_t = torch.tensor(mask, dtype=torch.float32, device=device).view(1, 1, H, W)

    x = torch.tensor(x0, dtype=torch.float32, device=device).view(1, 1, H, W)
    x.requires_grad_(True)

    ae = ae.to(device)
    ae.eval()

    for p in ae.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam([x], lr=lr)

    log = {
        "total_loss": [],
        "data_loss": [],
        "ae_loss": [],
        "tv_loss": [],
        "psnr": [],
        "ssim": [],
        "step": [],
    }

    if x_gt is not None:
        try:
            from .metrics import compute_psnr, compute_ssim
        except ImportError:
            from metrics import compute_psnr, compute_ssim
        x_gt = _to_float01(x_gt)

    for step in range(steps):
        optimizer.zero_grad()

        data_loss = 0.5 * ((m_t * x - y_t) ** 2).mean()

        ae_loss = ae_patch_loss(
            x,
            ae,
            patch_size=patch_size,
            stride=stride,
            ae_input_format=ae_input_format
        )

        tv_loss = torch_tv_loss(x)

        total_loss = data_loss + lam_ae * ae_loss + lam_tv * tv_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)

            if fix_observed:
                x.copy_(m_t * y_t + (1.0 - m_t) * x)

        log["total_loss"].append(float(total_loss.detach().cpu()))
        log["data_loss"].append(float(data_loss.detach().cpu()))
        log["ae_loss"].append(float(ae_loss.detach().cpu()))
        log["tv_loss"].append(float(tv_loss.detach().cpu()))

        if x_gt is not None and (step % log_interval == 0 or step == steps - 1):
            x_np = x.detach().cpu().numpy().reshape(H, W)
            psnr = compute_psnr(x_np, x_gt)
            ssim = compute_ssim(x_np, x_gt)

            log["step"].append(step)
            log["psnr"].append(float(psnr))
            log["ssim"].append(float(ssim))

        if verbose and (step % 100 == 0 or step == steps - 1):
            msg = (
                f"[TV+AE] step {step:04d} | "
                f"total={log['total_loss'][-1]:.6f} | "
                f"data={log['data_loss'][-1]:.6f} | "
                f"ae={log['ae_loss'][-1]:.6f} | "
                f"tv={log['tv_loss'][-1]:.6f}"
            )

            if x_gt is not None and len(log["psnr"]) > 0:
                msg += f" | psnr={log['psnr'][-1]:.4f}"

            print(msg)

    x_hat = x.detach().cpu().numpy().reshape(H, W)
    x_hat = _to_float01(x_hat)

    return x_hat, log
