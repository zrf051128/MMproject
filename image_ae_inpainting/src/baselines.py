import numpy as np


def _to_float01(x):
    x = np.asarray(x).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def _init_image(y, mask, init="telea"):
    if init == "telea":
        return opencv_telea(y, mask)
    if init == "ns":
        return opencv_ns(y, mask)
    if init == "zero":
        return y.copy()
    if init == "mean":
        x = y.copy()
        observed = y[mask == 1]
        mean_val = observed.mean() if observed.size > 0 else 0.0
        x[mask == 0] = mean_val
        return _to_float01(x)

    raise ValueError(f"Unknown init method: {init}")


# ============================================================
# Baseline 1: OpenCV Telea
# ============================================================

def opencv_telea(y, mask, radius=3):
    """
    OpenCV Telea inpainting.

    y: corrupted image, [H, W], range [0, 1]
    mask: 1 observed, 0 missing
    """
    import cv2

    y = _to_float01(y)
    mask = np.asarray(mask).astype(np.float32)

    img_uint8 = (y * 255.0).astype(np.uint8)

    # OpenCV requires missing region = 255, observed region = 0
    missing_uint8 = ((1.0 - mask) * 255.0).astype(np.uint8)

    out = cv2.inpaint(
        img_uint8,
        missing_uint8,
        radius,
        cv2.INPAINT_TELEA
    )

    return _to_float01(out.astype(np.float32) / 255.0)


# ============================================================
# Baseline 2: OpenCV Navier-Stokes
# ============================================================

def opencv_ns(y, mask, radius=3):
    """
    OpenCV Navier-Stokes inpainting.

    y: corrupted image, [H, W], range [0, 1]
    mask: 1 observed, 0 missing
    """
    import cv2

    y = _to_float01(y)
    mask = np.asarray(mask).astype(np.float32)

    img_uint8 = (y * 255.0).astype(np.uint8)

    # OpenCV requires missing region = 255, observed region = 0
    missing_uint8 = ((1.0 - mask) * 255.0).astype(np.uint8)

    out = cv2.inpaint(
        img_uint8,
        missing_uint8,
        radius,
        cv2.INPAINT_NS
    )

    return _to_float01(out.astype(np.float32) / 255.0)


# ============================================================
# Baseline 3: TV inpainting
# ============================================================

def _torch_tv_loss(x, eps=1e-6):
    """
    Isotropic TV loss for tensor x with shape [1, 1, H, W].
    """
    dx = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]
    dy = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]

    tv = torch.sqrt(dx ** 2 + dy ** 2 + eps).mean()
    return tv


def tv_inpainting(
    y,
    mask,
    lam_tv=1e-3,
    lr=1e-2,
    steps=800,
    init="telea",
    verbose=False,
):
    """
    TV inpainting baseline.

    Optimization model:

        min_x  0.5 || M ⊙ x - y ||_2^2 + lambda_TV TV(x)

    y: corrupted image, [H, W], range [0, 1]
    mask: 1 observed, 0 missing
    """
    global torch
    import torch

    y = _to_float01(y)
    mask = np.asarray(mask).astype(np.float32)

    H, W = y.shape

    x0 = _init_image(y, mask, init=init)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_t = torch.tensor(y, dtype=torch.float32, device=device).view(1, 1, H, W)
    m_t = torch.tensor(mask, dtype=torch.float32, device=device).view(1, 1, H, W)

    x = torch.tensor(x0, dtype=torch.float32, device=device).view(1, 1, H, W)
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    log = {
        "total_loss": [],
        "data_loss": [],
        "tv_loss": [],
    }

    for step in range(steps):
        optimizer.zero_grad()

        data_loss = 0.5 * ((m_t * x - y_t) ** 2).mean()
        tv_loss = _torch_tv_loss(x)

        total_loss = data_loss + lam_tv * tv_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)

        log["total_loss"].append(float(total_loss.detach().cpu()))
        log["data_loss"].append(float(data_loss.detach().cpu()))
        log["tv_loss"].append(float(tv_loss.detach().cpu()))

        if verbose and (step % 100 == 0 or step == steps - 1):
            print(
                f"[TV] step {step:04d} | "
                f"total={log['total_loss'][-1]:.6f} | "
                f"data={log['data_loss'][-1]:.6f} | "
                f"tv={log['tv_loss'][-1]:.6f}"
            )

    x_hat = x.detach().cpu().numpy().reshape(H, W)
    return _to_float01(x_hat), log


# ============================================================
# Baseline 4: Wavelet sparse inpainting
# ============================================================

def _soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _wavelet_soft_threshold_2d(image, wavelet="db2", level=2, threshold=0.01):
    """
    Apply wavelet soft-thresholding to a 2D image.
    """
    import pywt

    H, W = image.shape

    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    new_coeffs = [coeffs[0]]

    for detail in coeffs[1:]:
        cH, cV, cD = detail
        new_detail = (
            _soft_threshold(cH, threshold),
            _soft_threshold(cV, threshold),
            _soft_threshold(cD, threshold),
        )
        new_coeffs.append(new_detail)

    rec = pywt.waverec2(new_coeffs, wavelet=wavelet)

    rec = rec[:H, :W]

    return _to_float01(rec)


def wavelet_sparse_inpainting(
    y,
    mask,
    lam_wavelet=0.01,
    steps=150,
    step_size=1.0,
    wavelet="db2",
    level=2,
    init="telea",
    enforce_observed=True,
    verbose=False,
):
    """
    Wavelet sparse inpainting baseline.

    Approximate optimization model:

        min_x  0.5 || M ⊙ x - y ||_2^2 + lambda || W x ||_1

    It uses iterative gradient step on the data term
    plus wavelet soft-thresholding.
    """
    y = _to_float01(y)
    mask = np.asarray(mask).astype(np.float32)

    x = _init_image(y, mask, init=init)

    log = {
        "data_loss": [],
        "wavelet_threshold": [],
    }

    for step in range(steps):
        # Gradient of 0.5 || M*x - y ||^2 is M*(M*x - y)
        grad = mask * (mask * x - y)

        z = x - step_size * grad

        x = _wavelet_soft_threshold_2d(
            z,
            wavelet=wavelet,
            level=level,
            threshold=lam_wavelet * step_size
        )

        if enforce_observed:
            # Keep observed pixels close to measurement
            x = mask * y + (1.0 - mask) * x

        x = _to_float01(x)

        data_loss = 0.5 * np.mean((mask * x - y) ** 2)

        log["data_loss"].append(float(data_loss))
        log["wavelet_threshold"].append(float(lam_wavelet * step_size))

        if verbose and (step % 30 == 0 or step == steps - 1):
            print(
                f"[Wavelet] step {step:04d} | "
                f"data={data_loss:.6f}"
            )

    return _to_float01(x), log


# ============================================================
# Baseline 5: K-means patch prior
# ============================================================

def _extract_patches_np(image, patch_size=8, stride=4):
    """
    Extract overlapping patches from image.

    Return:
        patches: [N, patch_size * patch_size]
        positions: list of (top, left)
    """
    H, W = image.shape
    patches = []
    positions = []

    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            patch = image[top:top + patch_size, left:left + patch_size]
            patches.append(patch.reshape(-1))
            positions.append((top, left))

    patches = np.asarray(patches, dtype=np.float32)
    return patches, positions


def _reconstruct_from_patches_np(patches, positions, image_shape, patch_size=8):
    """
    Reconstruct image from overlapping patches by averaging.
    """
    H, W = image_shape

    image_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for patch_vec, (top, left) in zip(patches, positions):
        patch = patch_vec.reshape(patch_size, patch_size)

        image_sum[top:top + patch_size, left:left + patch_size] += patch
        weight_sum[top:top + patch_size, left:left + patch_size] += 1.0

    weight_sum = np.maximum(weight_sum, 1e-8)

    return image_sum / weight_sum


def kmeans_patch_prior_inpainting(
    y,
    mask,
    patch_size=8,
    stride=4,
    n_clusters=64,
    outer_iters=15,
    alpha=0.8,
    visible_threshold=0.6,
    init="telea",
    random_state=0,
    verbose=False,
):
    """
    K-means patch prior baseline.

    It learns patch centers using K-means, then iteratively replaces
    each patch by its nearest cluster center and averages overlapping patches.

    This is a patch-prior baseline, not your AE prior.

    y: corrupted image, [H, W], range [0, 1]
    mask: 1 observed, 0 missing
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    y = _to_float01(y)
    mask = np.asarray(mask).astype(np.float32)

    H, W = y.shape

    x = _init_image(y, mask, init=init)

    # Extract initial patches from initialized image
    init_patches, positions = _extract_patches_np(
        x,
        patch_size=patch_size,
        stride=stride
    )

    # Decide which patches are sufficiently visible
    mask_patches, _ = _extract_patches_np(
        mask,
        patch_size=patch_size,
        stride=stride
    )

    visible_ratio = mask_patches.mean(axis=1)
    selected = visible_ratio >= visible_threshold

    # If too few visible patches, fall back to all initialized patches
    if selected.sum() < max(10, n_clusters):
        train_patches = init_patches
    else:
        train_patches = init_patches[selected]

    n_clusters_eff = min(n_clusters, train_patches.shape[0])

    if n_clusters_eff < 2:
        return x, {"note": "Too few patches for K-means."}

    kmeans = KMeans(
        n_clusters=n_clusters_eff,
        random_state=random_state,
        n_init=10
    )

    kmeans.fit(train_patches)
    centers = kmeans.cluster_centers_.astype(np.float32)

    log = {
        "patch_error": [],
    }

    for it in range(outer_iters):
        patches, positions = _extract_patches_np(
            x,
            patch_size=patch_size,
            stride=stride
        )

        nearest_idx, distances = pairwise_distances_argmin_min(
            patches,
            centers
        )

        projected_patches = centers[nearest_idx]

        prior_image = _reconstruct_from_patches_np(
            projected_patches,
            positions,
            image_shape=(H, W),
            patch_size=patch_size
        )

        # Update only missing region more strongly
        x_new = mask * y + (1.0 - mask) * (
            (1.0 - alpha) * x + alpha * prior_image
        )

        x = _to_float01(x_new)

        patch_error = float(np.mean(distances))
        log["patch_error"].append(patch_error)

        if verbose:
            print(f"[KMeans Patch] iter {it:03d} | patch_error={patch_error:.6f}")

    return _to_float01(x), log


# ============================================================
# Unified interface
# ============================================================

def run_baseline(method_name, y, mask, **kwargs):
    """
    Unified interface.

    Supported method_name:
        - opencv_telea
        - opencv_ns
        - tv
        - wavelet
        - kmeans_patch
    """
    method_name = method_name.lower()

    if method_name == "opencv_telea":
        return opencv_telea(y, mask, **kwargs)

    if method_name == "opencv_ns":
        return opencv_ns(y, mask, **kwargs)

    if method_name == "tv":
        x_hat, log = tv_inpainting(y, mask, **kwargs)
        return x_hat

    if method_name == "wavelet":
        x_hat, log = wavelet_sparse_inpainting(y, mask, **kwargs)
        return x_hat

    if method_name == "kmeans_patch":
        x_hat, log = kmeans_patch_prior_inpainting(y, mask, **kwargs)
        return x_hat

    raise ValueError(f"Unknown baseline method: {method_name}")