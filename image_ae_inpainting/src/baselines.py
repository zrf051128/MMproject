"""
Optimization-based mathematical priors used as baselines.

All methods solve an image variable x with the same conventions:
  - mean-value initialization on missing pixels
  - Adam optimization
  - hard data-consistency projection after each step
"""

import numpy as np


_DCT_MATRIX_CACHE = {}


def _to_float01(x):
    x = np.asarray(x).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def mean_initialization(y, mask):
    """Initialize missing pixels with the mean of observed pixels."""
    y = _to_float01(y)
    mask = np.asarray(mask).astype(np.float32)
    observed = y[mask == 1]
    known_mean = float(observed.mean()) if observed.size > 0 else 0.0
    x0 = mask * y + (1.0 - mask) * known_mean
    return _to_float01(x0)


def _dct_matrix(n, dtype, device):
    """Orthonormal DCT-II transform matrix."""
    import torch

    key = (n, str(dtype), str(device))
    if key in _DCT_MATRIX_CACHE:
        return _DCT_MATRIX_CACHE[key]

    k = torch.arange(n, dtype=dtype, device=device).view(n, 1)
    i = torch.arange(n, dtype=dtype, device=device).view(1, n)
    mat = torch.cos(np.pi / n * (i + 0.5) * k)
    mat[0, :] *= np.sqrt(1.0 / n)
    if n > 1:
        mat[1:, :] *= np.sqrt(2.0 / n)

    _DCT_MATRIX_CACHE[key] = mat
    return mat


def dct_l1_loss(x):
    """DCT-only sparse prior: mean absolute DCT coefficients."""
    import torch

    b, c, h, w = x.shape
    dct_h = _dct_matrix(h, x.dtype, x.device)
    dct_w = _dct_matrix(w, x.dtype, x.device)

    x_flat = x.reshape(b * c, h, w)
    coeff = torch.matmul(dct_h, torch.matmul(x_flat, dct_w.t()))
    return coeff.abs().mean()


def haar_wavelet_l1_loss(x):
    """Differentiable one-level Haar high-frequency L1 penalty."""
    _, _, h, w = x.shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    x = x[:, :, :h_even, :w_even]

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    lh = (x00 - x01 + x10 - x11) / 4.0
    hl = (x00 + x01 - x10 - x11) / 4.0
    hh = (x00 - x01 - x10 + x11) / 4.0
    return lh.abs().mean() + hl.abs().mean() + hh.abs().mean()


def tv_loss(x, eps=1e-6):
    """Differentiable isotropic TV penalty."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]

    dx_crop = dx[:, :, :-1, :]
    dy_crop = dy[:, :, :, :-1]
    return (dx_crop.pow(2) + dy_crop.pow(2) + eps).sqrt().mean()


def _optimize_with_regularizer(
    y,
    mask,
    regularizer_fn,
    regularizer_name,
    lam,
    lr=0.03,
    steps=500,
    x_gt=None,
    log_interval=20,
    verbose=False,
):
    import torch

    y = _to_float01(y)
    mask = np.asarray(mask).astype(np.float32)
    h, w = y.shape

    x0 = mean_initialization(y, mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_t = torch.tensor(y, dtype=torch.float32, device=device).view(1, 1, h, w)
    m_t = torch.tensor(mask, dtype=torch.float32, device=device).view(1, 1, h, w)
    x = torch.tensor(x0, dtype=torch.float32, device=device).view(1, 1, h, w)
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    log = {
        "total_loss": [],
        "data_loss": [],
        regularizer_name: [],
        "step": [],
        "psnr": [],
        "ssim": [],
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
        reg_loss = regularizer_fn(x)
        total_loss = data_loss + lam * reg_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)
            x.copy_(m_t * y_t + (1.0 - m_t) * x)

        log["total_loss"].append(float(total_loss.detach().cpu()))
        log["data_loss"].append(float(data_loss.detach().cpu()))
        log[regularizer_name].append(float(reg_loss.detach().cpu()))

        if x_gt is not None and (step % log_interval == 0 or step == steps - 1):
            x_np = x.detach().cpu().numpy().reshape(h, w)
            log["step"].append(step)
            log["psnr"].append(float(compute_psnr(x_np, x_gt)))
            log["ssim"].append(float(compute_ssim(x_np, x_gt)))

        if verbose and (step % 100 == 0 or step == steps - 1):
            print(
                f"[{regularizer_name}] step {step:04d} | "
                f"total={log['total_loss'][-1]:.6f} | "
                f"data={log['data_loss'][-1]:.6f} | "
                f"reg={log[regularizer_name][-1]:.6f}"
            )

    x_hat = x.detach().cpu().numpy().reshape(h, w)
    return _to_float01(x_hat), log


def dct_inpainting(
    y,
    mask,
    lam_dct=0.01,
    lr=0.03,
    steps=500,
    x_gt=None,
    log_interval=20,
    verbose=False,
):
    return _optimize_with_regularizer(
        y,
        mask,
        regularizer_fn=dct_l1_loss,
        regularizer_name="dct_loss",
        lam=lam_dct,
        lr=lr,
        steps=steps,
        x_gt=x_gt,
        log_interval=log_interval,
        verbose=verbose,
    )


def wavelet_inpainting(
    y,
    mask,
    lam_wavelet=0.01,
    lr=0.03,
    steps=500,
    x_gt=None,
    log_interval=20,
    verbose=False,
):
    return _optimize_with_regularizer(
        y,
        mask,
        regularizer_fn=haar_wavelet_l1_loss,
        regularizer_name="wavelet_loss",
        lam=lam_wavelet,
        lr=lr,
        steps=steps,
        x_gt=x_gt,
        log_interval=log_interval,
        verbose=verbose,
    )


def tv_inpainting(
    y,
    mask,
    lam_tv=0.01,
    lr=0.03,
    steps=500,
    x_gt=None,
    log_interval=20,
    verbose=False,
):
    return _optimize_with_regularizer(
        y,
        mask,
        regularizer_fn=tv_loss,
        regularizer_name="tv_loss",
        lam=lam_tv,
        lr=lr,
        steps=steps,
        x_gt=x_gt,
        log_interval=log_interval,
        verbose=verbose,
    )


def run_baseline(method_name, y, mask, **kwargs):
    """Run one mathematical-prior baseline."""
    method_name = method_name.lower()

    if method_name in {"dct", "dct_only"}:
        x_hat, _log = dct_inpainting(y, mask, **kwargs)
        return x_hat

    if method_name == "wavelet":
        x_hat, _log = wavelet_inpainting(y, mask, **kwargs)
        return x_hat

    if method_name in {"tv", "tv_only"}:
        x_hat, _log = tv_inpainting(y, mask, **kwargs)
        return x_hat

    raise ValueError(f"Unknown baseline method: {method_name}")
