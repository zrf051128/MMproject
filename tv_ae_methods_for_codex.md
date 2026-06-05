# Optimization-Based Image Inpainting Methods for TV+AE Experiments

## 0. Goal

This document defines the methods that should be implemented for comparing pure mathematical optimization models with the proposed neural-prior-enhanced optimization model.

The main purpose is to verify whether adding an autoencoder-based patch prior improves traditional variational image inpainting.

The core comparison is:

\[
\text{Pure mathematical priors}
\quad \text{vs.} \quad
\text{TV regularization + learned AE patch prior}
\]

---

## 1. Common Problem Setting

### 1.1 Variables

Let:

\[
x \in [0,1]^{H \times W}
\]

be the restored image to be optimized.

Let:

\[
x_{\text{gt}} \in [0,1]^{H \times W}
\]

be the ground-truth clean image, used only for evaluation.

Let:

\[
M \in \{0,1\}^{H \times W}
\]

be the binary mask, where:

\[
M_{ij}=1
\quad \Rightarrow \quad
\text{pixel } (i,j) \text{ is observed}
\]

\[
M_{ij}=0
\quad \Rightarrow \quad
\text{pixel } (i,j) \text{ is missing}
\]

Let:

\[
y = M \odot x_{\text{gt}}
\]

be the corrupted image.

Here, \(\odot\) denotes element-wise multiplication.

---

### 1.2 Data Fidelity Term

All optimization-based methods should use the same data fidelity term:

\[
D(x)
=
\frac{1}{2}
\|M \odot x - y\|_2^2
\]

This term forces the restored image \(x\) to match the known pixels in \(y\).

---

### 1.3 Hard Data Consistency Projection

After each optimization update, apply:

\[
x \leftarrow M \odot y + (1-M)\odot x
\]

This ensures that the known pixels remain unchanged during optimization.

In code:

```python
x.data = mask * y + (1 - mask) * x.data
```

or, safer:

```python
with torch.no_grad():
    x.clamp_(0.0, 1.0)
    x.copy_(mask * y + (1.0 - mask) * x)
```

---

### 1.4 Initialization

Use the same initialization for all optimization-based methods.

Recommended initialization:

\[
x^0 = M\odot y + (1-M)\odot \bar{y}
\]

where:

\[
\bar{y}
=
\frac{\sum_{i,j} M_{ij}y_{ij}}{\sum_{i,j}M_{ij}}
\]

is the mean value of the known pixels.

In code:

```python
known_mean = y[mask == 1].mean()
x0 = mask * y + (1 - mask) * known_mean
```

This initialization is only used as the starting point. It is not a separate baseline in the main experiment.

---

## 2. Method 1: Laplacian / Tikhonov Regularization

### 2.1 Purpose

This is the simplest pure mathematical optimization model.

It represents a smoothness prior and is used to show that simple smooth optimization tends to oversmooth image structures.

---

### 2.2 Objective Function

\[
\hat{x}
=
\arg\min_x
\frac{1}{2}
\|M\odot x-y\|_2^2
+
\lambda_L
\|\nabla x\|_2^2
\]

where:

\[
\nabla x = (\nabla_h x, \nabla_v x)
\]

is the discrete image gradient.

The regularization term can be written as:

\[
R_L(x)
=
\|\nabla x\|_2^2
=
\sum_{i,j}
\left[
(x_{i,j+1}-x_{i,j})^2
+
(x_{i+1,j}-x_{i,j})^2
\right]
\]

Use valid differences or padding consistently.

---

### 2.3 Loss in Code

```python
def laplacian_smoothness_loss(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()
```

Full loss:

```python
loss = data_loss + lambda_lap * laplacian_smoothness_loss(x)
```

---

### 2.4 Suggested Method Name

Use one of:

```text
Laplacian
```

or:

```text
Tikhonov
```

Keep the name consistent in all CSV files and plots.

---

## 3. Method 2: Wavelet Sparsity Regularization

### 3.1 Purpose

This is a traditional mathematical optimization model based on sparse representation.

It is used to compare the proposed learned AE prior with a hand-crafted sparse prior.

---

### 3.2 Objective Function

\[
\hat{x}
=
\arg\min_x
\frac{1}{2}
\|M\odot x-y\|_2^2
+
\lambda_W
\|Bx\|_1
\]

where:

\[
B
\]

is a wavelet transform operator.

\[
\|Bx\|_1
\]

encourages sparsity of the image in the wavelet domain.

---

### 3.3 Practical Implementation Options

There are two acceptable implementation choices.

#### Option A: Use PyWavelets outside autograd

Use iterative shrinkage / thresholding style updates.

This is more classical but more coding work.

#### Option B: Use a differentiable Haar-like transform in PyTorch

Implement a simple differentiable Haar wavelet penalty.

For a simple experiment, implement horizontal, vertical, and diagonal high-frequency components using average/difference pooling.

Example simplified Haar penalty:

```python
def haar_wavelet_l1_loss(x):
    # x shape: [B, C, H, W]
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) / 4.0
    lh = (x00 - x01 + x10 - x11) / 4.0
    hl = (x00 + x01 - x10 - x11) / 4.0
    hh = (x00 - x01 - x10 + x11) / 4.0

    return lh.abs().mean() + hl.abs().mean() + hh.abs().mean()
```

Full loss:

```python
loss = data_loss + lambda_wavelet * haar_wavelet_l1_loss(x)
```

---

### 3.4 Suggested Method Name

```text
Wavelet
```

---

## 4. Method 3: TV-Only Regularization

### 4.1 Purpose

This is the strongest pure mathematical optimization baseline.

It is the most important baseline because the proposed method is TV+AE.

The core comparison is:

\[
\text{TV-only}
\quad \text{vs.} \quad
\text{TV+AE}
\]

---

### 4.2 Objective Function

\[
\hat{x}
=
\arg\min_x
\frac{1}{2}
\|M\odot x-y\|_2^2
+
\lambda_{TV}
R_{TV}(x)
\]

Use the differentiable isotropic TV form:

\[
R_{TV}(x)
=
\sum_{i,j}
\sqrt{
(x_{i,j+1}-x_{i,j})^2
+
(x_{i+1,j}-x_{i,j})^2
+
\epsilon
}
\]

where:

\[
\epsilon > 0
\]

is a small constant for numerical stability.

Recommended:

\[
\epsilon = 10^{-6}
\]

---

### 4.3 Loss in Code

```python
def tv_loss(x, eps=1e-6):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]

    # Match shapes
    dx_crop = dx[:, :, :-1, :]
    dy_crop = dy[:, :, :, :-1]

    return torch.sqrt(dx_crop.pow(2) + dy_crop.pow(2) + eps).mean()
```

Full loss:

```python
loss = data_loss + lambda_tv * tv_loss(x)
```

---

### 4.4 Suggested Method Name

```text
TV
```

or:

```text
TV-only
```

Use one consistent name.

---

## 5. Method 4: AE-Only Regularization

### 5.1 Purpose

This is the neural-prior ablation.

It tests whether the learned patch autoencoder prior alone can help inpainting.

This method is necessary because it shows the effect of the AE prior without TV.

---

### 5.2 Patch Operator

Let:

\[
P_i x \in \mathbb{R}^{s^2}
\]

be the \(i\)-th image patch extracted from \(x\).

For grayscale image patches:

\[
s=8
\]

so:

\[
P_i x \in \mathbb{R}^{64}
\]

Use overlapping patches.

Recommended:

\[
s=8
\]

\[
\text{stride}=4
\]

---

### 5.3 Autoencoder

Let:

\[
A_\theta(p)
\]

be a trained patch autoencoder.

Recommended architecture:

\[
64 \rightarrow 32 \rightarrow d \rightarrow 32 \rightarrow 64
\]

where:

\[
d \in \{8,16,32\}
\]

Recommended default:

\[
d=16
\]

The autoencoder is trained before image optimization.

During image optimization, the AE parameters are fixed.

---

### 5.4 AE Training Loss

Extract highly visible patches from the corrupted image.

For patch \(P_i y\), define visible ratio:

\[
\rho_i
=
\frac{\|P_i M\|_0}{s^2}
\]

Use only patches satisfying:

\[
\rho_i \ge r
\]

Recommended:

\[
r=0.8
\]

The training patch set is:

\[
\mathcal{P}_{vis}
=
\{P_i y: \rho_i \ge r\}
\]

Train the autoencoder by:

\[
\theta^\star
=
\arg\min_\theta
\frac{1}{|\mathcal{P}_{vis}|}
\sum_{p_i\in \mathcal{P}_{vis}}
\|p_i-A_\theta(p_i)\|_2^2
\]

---

### 5.5 AE Regularization Term

After training, freeze the autoencoder.

Define:

\[
R_{AE}(x)
=
\frac{1}{|\mathcal{I}|}
\sum_{i\in\mathcal{I}}
\frac{1}{s^2}
\|P_i x-A_{\theta^\star}(P_i x)\|_2^2
\]

where:

\[
\mathcal{I}
\]

is the set of all patch locations.

The normalization by \(|\mathcal{I}|\) and \(s^2\) is important to make the loss scale stable.

---

### 5.6 Objective Function

\[
\hat{x}
=
\arg\min_x
\frac{1}{2}
\|M\odot x-y\|_2^2
+
\lambda_{AE}
R_{AE}(x)
\]

---

### 5.7 Loss in Code

Use `torch.nn.Unfold` to extract overlapping patches.

```python
def extract_patches_torch(x, patch_size=8, stride=4):
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    patches = unfold(x)  # [B, C*patch_size*patch_size, L]
    patches = patches.transpose(1, 2)  # [B, L, C*patch_size*patch_size]
    return patches
```

AE regularization:

```python
def ae_regularization_loss(x, ae, patch_size=8, stride=4):
    patches = extract_patches_torch(x, patch_size=patch_size, stride=stride)
    B, L, D = patches.shape

    patches_flat = patches.reshape(B * L, D)
    recon_flat = ae(patches_flat)

    loss = (patches_flat - recon_flat).pow(2).mean()
    return loss
```

Full loss:

```python
loss = data_loss + lambda_ae * ae_regularization_loss(x, ae)
```

---

### 5.8 Suggested Method Name

```text
AE-only
```

---

## 6. Method 5: Proposed TV+AE Method

### 6.1 Purpose

This is the proposed method.

It combines:

1. TV regularization, which provides geometric smoothness and edge-preserving constraints.
2. AE patch prior, which provides learned image-specific patch-level information.

This method is used to prove that:

\[
\text{pure mathematical optimization prior}
\]

is not enough, and adding a learned neural patch prior can improve image inpainting.

---

### 6.2 Objective Function

\[
\hat{x}
=
\arg\min_x
\frac{1}{2}
\|M\odot x-y\|_2^2
+
\lambda_{TV}
R_{TV}(x)
+
\lambda_{AE}
R_{AE}(x)
\]

where:

\[
R_{TV}(x)
=
\sum_{i,j}
\sqrt{
(x_{i,j+1}-x_{i,j})^2
+
(x_{i+1,j}-x_{i,j})^2
+
\epsilon
}
\]

and:

\[
R_{AE}(x)
=
\frac{1}{|\mathcal{I}|}
\sum_{i\in\mathcal{I}}
\frac{1}{s^2}
\|P_i x-A_{\theta^\star}(P_i x)\|_2^2
\]

---

### 6.3 Loss in Code

```python
data_loss = 0.5 * ((mask * x - y) ** 2).mean()

loss = (
    data_loss
    + lambda_tv * tv_loss(x)
    + lambda_ae * ae_regularization_loss(x, ae, patch_size=8, stride=4)
)
```

After each optimizer step:

```python
with torch.no_grad():
    x.clamp_(0.0, 1.0)
    x.copy_(mask * y + (1.0 - mask) * x)
```

---

### 6.4 Suggested Method Name

Use:

```text
TV+AE
```

or:

```text
ours
```

Recommended:

```text
TV+AE
```

because it is clearer for plots and tables.

---

## 7. Unified Optimization Procedure

Use the same optimization loop for all methods.

### 7.1 Optimizer

Use Adam.

Recommended default:

```python
lr = 0.03
steps = 500
```

Possible values to tune:

```python
lr in [0.01, 0.03, 0.05]
steps in [300, 500, 800]
```

---

### 7.2 Generic Optimization Loop

```python
def optimize_image(
    y,
    mask,
    method,
    ae=None,
    lambda_lap=0.01,
    lambda_wavelet=0.01,
    lambda_tv=0.01,
    lambda_ae=0.01,
    lr=0.03,
    steps=500,
):
    # y, mask shape: [1, 1, H, W]
    known_mean = y[mask == 1].mean()
    x0 = mask * y + (1.0 - mask) * known_mean

    x = x0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()

        data_loss = 0.5 * ((mask * x - y) ** 2).mean()
        reg_loss = 0.0

        if method == "laplacian":
            reg_loss = lambda_lap * laplacian_smoothness_loss(x)

        elif method == "wavelet":
            reg_loss = lambda_wavelet * haar_wavelet_l1_loss(x)

        elif method == "tv":
            reg_loss = lambda_tv * tv_loss(x)

        elif method == "ae":
            reg_loss = lambda_ae * ae_regularization_loss(x, ae)

        elif method == "tv_ae":
            reg_loss = (
                lambda_tv * tv_loss(x)
                + lambda_ae * ae_regularization_loss(x, ae)
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        loss = data_loss + reg_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)
            x.copy_(mask * y + (1.0 - mask) * x)

        loss_history.append(float(loss.detach().cpu()))

    return x.detach(), loss_history
```

---

## 8. Required Methods in Final Experiment

The final experiment should include exactly these five methods:

| Method Name | Model Type | Objective |
|---|---|---|
| Laplacian | Pure mathematical optimization | \(D(x)+\lambda_L\|\nabla x\|_2^2\) |
| Wavelet | Pure mathematical optimization | \(D(x)+\lambda_W\|Bx\|_1\) |
| TV | Pure mathematical optimization | \(D(x)+\lambda_{TV}R_{TV}(x)\) |
| AE-only | Neural-prior optimization | \(D(x)+\lambda_{AE}R_{AE}(x)\) |
| TV+AE | Proposed method | \(D(x)+\lambda_{TV}R_{TV}(x)+\lambda_{AE}R_{AE}(x)\) |

---

## 9. Suggested Masks

Do not use random masks as the main experiment.

Use structured missing-region masks:

1. Block mask
2. Irregular mask
3. Scratch / line mask
4. Text mask

The main reason is that structured masks better reflect missing-region inpainting, while random masks are often favorable to local smoothing methods.

---

## 10. Evaluation Metrics

Compute both full-image metrics and missing-region-only metrics.

### 10.1 Full Image Metrics

Compute:

\[
\text{PSNR}
\]

\[
\text{SSIM}
\]

\[
\text{RMSE}
\]

\[
\text{MAE}
\]

on the whole image.

---

### 10.2 Missing-Region-Only Metrics

This is more important for inpainting.

Let:

\[
\bar{M}=1-M
\]

Then compute errors only on missing pixels.

For example:

\[
\text{MSE}_{miss}
=
\frac{
\sum_{i,j}(1-M_{ij})(x_{ij}-x_{\text{gt},ij})^2
}{
\sum_{i,j}(1-M_{ij})
}
\]

\[
\text{RMSE}_{miss}
=
\sqrt{\text{MSE}_{miss}}
\]

\[
\text{MAE}_{miss}
=
\frac{
\sum_{i,j}(1-M_{ij})|x_{ij}-x_{\text{gt},ij}|
}{
\sum_{i,j}(1-M_{ij})
}
\]

\[
\text{PSNR}_{miss}
=
10\log_{10}
\frac{1}{\text{MSE}_{miss}}
\]

assuming image values are normalized to \([0,1]\).

---



## 12. Recommended CSV Columns

Each experiment result should be saved with these columns:

```text
image
mask_type
method
lambda_lap
lambda_wavelet
lambda_tv
lambda_ae
patch_size
stride
latent_dim
psnr
ssim
rmse
mae
psnr_missing
rmse_missing
mae_missing
runtime
```

Optional:

```text
ae_train_loss
final_total_loss
final_data_loss
final_tv_loss
final_ae_loss
```


---

## 14. Implementation Priority for Codex

Implement in this order:

1. `tv_loss`
2. `laplacian_smoothness_loss`
3. `haar_wavelet_l1_loss`
4. `extract_patches_torch`
5. `ae_regularization_loss`
6. `train_patch_autoencoder`
7. `optimize_image`
8. `compute_metrics`
9. `run_experiments`
10. `plot_results`

---

## 15. Important Sanity Checks

### Check 1: TV+AE with \(\lambda_{AE}=0\)

When:

\[
\lambda_{AE}=0
\]

TV+AE should produce almost the same result as TV-only.

If not, the implementation is inconsistent.

---

### Check 2: AE-only with \(\lambda_{AE}=0\)

When:

\[
\lambda_{AE}=0
\]

AE-only becomes data fidelity only.

The missing region should mostly remain close to initialization.

---

### Check 3: Known pixels must remain unchanged

After optimization:

\[
M\odot \hat{x}
=
M\odot y
\]

Check numerically:

```python
known_error = ((mask * x_hat - mask * y).abs()).max()
```

This value should be close to zero.

---

### Check 4: Loss Scale

Use mean losses instead of sum losses.

Recommended:

```python
data_loss = ((mask * x - y) ** 2).mean()
tv = tv_loss(x)
ae = ae_regularization_loss(x, ae)
```

Avoid unnormalized sums, because AE patch loss can become too large when many overlapping patches are used.

---

## 16. Hyperparameter Defaults

Use these as initial values:

```python
lambda_lap = 0.02
lambda_wavelet = 0.01
lambda_tv = 0.01
lambda_ae = 0.01

patch_size = 8
stride = 4
latent_dim = 16

ae_epochs = 50
ae_lr = 1e-3
ae_batch_size = 128

image_optimization_steps = 500
image_optimization_lr = 0.03
```

Then tune:

```python
lambda_tv in [0.003, 0.01, 0.03, 0.1]
lambda_ae in [0.001, 0.003, 0.01, 0.03, 0.1]
latent_dim in [8, 16, 32]
patch_size in [4, 8, 12]
```

---

## 17. Final Main Comparison

The main result table should compare:

```text
Laplacian
Wavelet
TV
AE-only
TV+AE
```

The key conclusion should focus on:

```text
TV vs TV+AE
AE-only vs TV+AE
Laplacian/Wavelet/TV vs TV+AE
```
