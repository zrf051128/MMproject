# 角色 B 后续执行步骤清单：Image AE Inpainting 实验主流程

本文档整理的是你作为 **角色 B** 在拿到 A 提供的图像数据、AE 代码和 AE 权重之后，接下来应该一步一步完成的工作。

你的核心职责是：

> 验证数据和权重 → 跑通 baseline → 接入 AE prior → 实现 Ours → 跑主实验 → 做 B 负责的消融实验 → 整理图表。

---

## 0. 当前已有内容说明

你现在已经有三类内容：

### 0.1 输入图像数据

路径建议为：

```text
MMPROJECT/data/images/
```

例如：

```text
MMPROJECT/data/images/barbara.png
MMPROJECT/data/images/house.png
MMPROJECT/data/images/peppers.png
MMPROJECT/data/images/lena.png
```

这些图像是实验中的 ground-truth images。后续会通过 mask 生成 corrupted image：

\[
y = M \odot x^\star
\]

其中：

- \(x^\star\)：原始完整图像；
- \(M\)：mask；
- \(y\)：缺失图像。

---

### 0.2 A 提供的 AE 代码

路径建议为：

```text
MMPROJECT/image_ae_inpainting/src/
```

包括：

```text
autoencoder.py
data_loader.py
patch_utils.py
train_ae.py
```

这些文件主要负责：

| 文件 | 作用 |
|---|---|
| `autoencoder.py` | 定义 tiny patch autoencoder |
| `data_loader.py` | 加载图像、转灰度、resize 到 128×128、归一化到 [0,1] |
| `patch_utils.py` | 提取 patches、筛选 visible patches |
| `train_ae.py` | 训练或加载 AE 权重 |

---

### 0.3 A 已经训练好的 AE 权重

路径建议为：

```text
MMPROJECT/results/ae_weights/
```

例如：

```text
barbara_p8_d16_current.pt
cameraman_p8_d16_current.pt
house_p8_d16_current.pt
lena_p8_d16_current.pt
peppers_p8_d16_current.pt
```

文件名含义：

```text
图像名_patch size_latent dim_training mode.pt
```

例如：

```text
cameraman_p8_d16_current.pt
```

表示：

- 图像：cameraman；
- patch size：8×8；
- latent dimension：16；
- training mode：current-image training。

这些权重可以先用于调试 Ours pipeline。

---

## 1. 项目目录结构确认

推荐保留你现在的结构：

```text
MMPROJECT/
│
├─ data/
│  ├─ images/
│  │  ├─ barbara.png
│  │  ├─ house.png
│  │  ├─ peppers.png
│  │  └─ lena.png
│  │
│  └─ bsd68/
│
├─ image_ae_inpainting/
│  ├─ scripts/
│  │  ├─ check_mask.py
│  │  ├─ check_metrics.py
│  │  ├─ check_baselines.py
│  │  ├─ check_optimization.py
│  │  ├─ check_all_method.py
│  │  ├─ check_ablation.py
│  │  ├─ run_main_experiments.py
│  │  └─ plot_results.py
│  │
│  └─ src/
│     ├─ autoencoder.py
│     ├─ baselines.py
│     ├─ data_loader.py
│     ├─ mask_generator.py
│     ├─ metrics.py
│     ├─ optimization.py
│     ├─ patch_utils.py
│     └─ train_ae.py
│
└─ results/
   ├─ ae_weights/
   ├─ restored_images/
   ├─ logs/
   ├─ tables/
   ├─ figures/
   ├─ debug_baselines/
   ├─ debug_mask/
   └─ debug_optimization/
```

---

## 2. 运行路径要求

运行所有脚本时，建议始终从项目根目录运行：

```powershell
cd D:\pythonProject\MMproject
```

然后运行：

```powershell
python image_ae_inpainting\scripts\check_mask.py
python image_ae_inpainting\scripts\check_baselines.py
python image_ae_inpainting\scripts\check_optimization.py
```

不要进入 `scripts/` 文件夹后再运行，否则相对路径可能找不到：

```text
./data/images/
./results/ae_weights/
```

---


---

# Part I：基础检查阶段

---

## 4. Step 1：检查图像能不能正确加载

### 4.1 目标

确认：

- 图像路径正确；
- 图像能够被 `data_loader.py` 加载；
- 图像大小统一为 \(128\times128\)；
- 像素范围是 \([0,1]\)；
- 数据类型是 `float32`。

### 4.2 建议脚本

可以写一个 `check_data.py`，或者直接在 `check_metrics.py` 中检查：

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "image_ae_inpainting" / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import load_all_images

images = load_all_images(image_size=128)

print(images.keys())

for name, img in images.items():
    print(name, img.shape, img.min(), img.max(), img.dtype)
```

### 4.3 预期结果

你应该看到：

```text
cameraman
barbara
house
peppers
lena
```

每张图满足：

```text
shape = (128, 128)
min >= 0
max <= 1
dtype = float32
```

### 4.4 注意事项

如果 `lena` 实际加载成了 `astronaut`，建议你手动放入真正的：

```text
data/images/lena.png
```

否则论文里写 Lena，但实际图像不是 Lena，会不严谨。

---

## 5. Step 2：检查 mask 是否正确生成

### 5.1 目标

你需要确认三种 mask 都正常：

1. random mask；
2. block mask；
3. irregular mask。

统一定义：

\[
M_{ij}=1
\]

表示 observed pixel；

\[
M_{ij}=0
\]

表示 missing pixel。

corrupted image 定义为：

\[
y = M \odot x^\star
\]

### 5.2 你需要检查的内容

- random 30%、50%、70% 的缺失比例是否正确；
- block mask 是否真的遮住连续区域；
- irregular mask 是否是随机线条/不规则形状；
- mask 有没有反；
- corrupted image 是否正确。

### 5.3 输出建议

保存到：

```text
results/debug_mask/
```

例如：

```text
cameraman_original.png
cameraman_random30_mask.png
cameraman_random30_corrupted.png
cameraman_random50_mask.png
cameraman_random50_corrupted.png
cameraman_block_mask.png
cameraman_block_corrupted.png
cameraman_irregular_mask.png
cameraman_irregular_corrupted.png
```

### 5.4 检查重点

如果 mask 反了，后面的所有 baseline 和 Ours 都会错。

正确情况应该是：

```python
corrupted = image * mask
```

---

## 6. Step 3：检查 metrics 是否正常

### 6.1 目标

确认 PSNR、SSIM、RMSE、MAE 计算正确。

### 6.2 需要实现的指标

```python
compute_psnr(x_hat, x_gt)
compute_ssim(x_hat, x_gt)
compute_rmse(x_hat, x_gt)
compute_mae(x_hat, x_gt)
```

### 6.3 检查方式

用同一张图对比自己：

```python
psnr = compute_psnr(img, img)
ssim = compute_ssim(img, img)
rmse = compute_rmse(img, img)
mae = compute_mae(img, img)
```

预期：

```text
PSNR = inf 或很大
SSIM = 1
RMSE = 0
MAE = 0
```

再用 corrupted image 对比 original image，看指标是否明显下降。

---

## 7. Step 4：检查 AE 权重能不能加载

### 7.1 目标

确认 A 给的 `.pt` 权重可以正常加载。

### 7.2 示例代码

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "image_ae_inpainting" / "src"
sys.path.insert(0, str(SRC_DIR))

from train_ae import load_autoencoder

ae_path = PROJECT_ROOT / "results" / "ae_weights" / "cameraman_p8_d16_current.pt"

ae = load_autoencoder(
    str(ae_path),
    patch_size=8,
    latent_dim=16
)

print("AE loaded successfully.")
```

### 7.3 注意事项

文件名和加载参数必须一致。

例如：

```text
cameraman_p8_d16_current.pt
```

必须对应：

```python
patch_size=8
latent_dim=16
```

如果是：

```text
cameraman_p12_d16_current.pt
```

则必须：

```python
patch_size=12
latent_dim=16
```

---

# Part II：Baseline 检查与调试阶段

---

## 8. Step 5：跑通 5 个 baseline

你已经决定只保留 5 个 baseline：

1. OpenCV Telea；
2. OpenCV Navier-Stokes；
3. TV inpainting；
4. Wavelet sparse inpainting；
5. K-means patch prior。

---

## 9. Baseline 1：OpenCV Telea

### 9.1 类型

经典快速图像修复算法。

### 9.2 作用

代表 fast classical inpainting method。

### 9.3 代码调用

```python
cv2.inpaint(img_uint8, mask_uint8, radius, cv2.INPAINT_TELEA)
```

注意：

- OpenCV 输入图像一般是 `uint8`，范围 \([0,255]\)；
- OpenCV 的 mask 中，缺失区域通常为 255；
- 你的 mask 是 \(1=\) observed, \(0=\) missing，所以要转换。

示例：

```python
img_uint8 = (y * 255).astype("uint8")
missing_uint8 = ((1 - mask) * 255).astype("uint8")
out = cv2.inpaint(img_uint8, missing_uint8, 3, cv2.INPAINT_TELEA)
out = out.astype("float32") / 255.0
```

---

## 10. Baseline 2：OpenCV Navier-Stokes

### 10.1 类型

PDE-inspired classical inpainting method。

### 10.2 作用

代表 PDE / structure propagation 类图像修复方法。

### 10.3 代码调用

```python
cv2.inpaint(img_uint8, mask_uint8, radius, cv2.INPAINT_NS)
```

输入格式和 Telea 一样。

---

## 11. Baseline 3：TV inpainting

### 11.1 类型

标准 variational optimization baseline。

### 11.2 数学模型

\[
\hat{x}_{TV}
=
\arg\min_x
\frac{1}{2}\|M\odot x-y\|_2^2
+
\lambda_{TV}\mathrm{TV}(x)
\]

其中：

\[
\mathrm{TV}(x)
=
\sum_{i,j}
\sqrt{
(x_{i+1,j}-x_{i,j})^2
+
(x_{i,j+1}-x_{i,j})^2
+
\epsilon
}
\]

### 11.3 作用

证明：

> 只靠 smoothness prior 不够，因此需要 learned AE patch prior。

### 11.4 建议参数

```text
lambda_TV = 1e-3
lr = 1e-2
steps = 500 或 1000
epsilon = 1e-6
```

---

## 12. Baseline 4：Wavelet sparse inpainting

### 12.1 类型

标准 sparse regularization optimization baseline。

### 12.2 数学模型

\[
\hat{x}_{W}
=
\arg\min_x
\frac{1}{2}\|M\odot x-y\|_2^2
+
\lambda_W \|Wx\|_1
\]

其中：

- \(W\)：wavelet transform；
- \(\|Wx\|_1\)：wavelet 域稀疏正则项。

### 12.3 作用

证明：

> 固定 sparse prior 不如 learned image-specific patch prior 灵活。

### 12.4 建议参数

```text
lambda_W = 1e-3 或 1e-2
steps = 500 或 1000
```

---

## 13. Baseline 5：K-means patch prior

### 13.1 类型

Patch prior optimization baseline。

### 13.2 数学模型

\[
\hat{x}_{KM}
=
\arg\min_x
\frac{1}{2}\|M\odot x-y\|_2^2
+
\lambda_{KM}
\sum_i
\min_k
\|P_i x-c_k\|_2^2
\]

其中：

- \(P_i x\)：第 \(i\) 个 patch；
- \(c_k\)：visible patches 聚类得到的第 \(k\) 个 cluster center。

### 13.3 作用

证明：

> AE patch manifold prior 比离散 K-means cluster prior 更灵活。

### 13.4 建议参数

```text
patch_size = 8
stride = 4
K = 64 或 128
lambda_KM = 1e-2
steps = 500 或 1000
```

---

## 14. Step 6：写 `check_baselines.py`

### 14.1 目标

先只用一张图、一个 mask 检查所有 baseline。

建议：

```text
image = cameraman
mask = random30 或 random50
size = 128×128
```

### 14.2 需要跑的方法

```text
OpenCV Telea
OpenCV Navier-Stokes
TV inpainting
Wavelet sparse inpainting
K-means patch prior
```

### 14.3 输出路径

```text
results/debug_baselines/
```

### 14.4 输出文件

```text
cameraman_random30_original.png
cameraman_random30_corrupted.png
cameraman_random30_telea.png
cameraman_random30_ns.png
cameraman_random30_tv.png
cameraman_random30_wavelet.png
cameraman_random30_kmeans.png
baseline_check.csv
```

### 14.5 `baseline_check.csv` 内容

```text
method, psnr, ssim, rmse, mae, runtime
```

### 14.6 检查重点

- 输出图像有没有全黑/全白；
- PSNR/SSIM 是否正常；
- TV / Wavelet / K-means 是否发散；
- 可见区域是否被严重改变；
- runtime 是否合理。

---

# Part III：接入 AE prior 并实现 Ours

---

## 15. Step 7：在 `optimization.py` 中实现 PyTorch 版 AE prior loss

### 15.1 为什么必须自己写 PyTorch 版

A 的 `patch_utils.py` 中 patch extraction 是 numpy 版。  
numpy 版本可以用于检查和 AE 训练，但不能用于优化图像变量 \(x\)，因为它不能反向传播。

你的 Ours 需要优化：

\[
x
\]

所以 AE prior loss 必须是 PyTorch 可求导版本。

---

## 16. PyTorch 版 AE prior loss

### 16.1 数学公式

\[
R_{AE}(x)
=
\sum_i
\|P_i x - A_\theta(P_i x)\|_2^2
\]

### 16.2 推荐代码

```python
import torch
import torch.nn.functional as F

def ae_prior_loss_torch(x, ae, patch_size=8, stride=4, reduction="mean"):
    patches = F.unfold(
        x,
        kernel_size=patch_size,
        stride=stride
    )  # [1, p*p, N]

    patches = patches.squeeze(0).transpose(0, 1)  # [N, p*p]

    recon = ae(patches)

    loss = (patches - recon) ** 2

    if reduction == "sum":
        return loss.sum()
    else:
        return loss.mean()
```

建议代码里用 `mean`，这样数值更稳定；论文公式仍然可以写 sum，因为 \(\lambda_{AE}\) 可以吸收尺度差异。

---

## 17. Step 8：实现完整 Ours 优化函数

### 17.1 数学模型

\[
\hat{x}
=
\arg\min_x
\frac{1}{2}\|M\odot x-y\|_2^2
+
\lambda_{AE}
\sum_i
\|P_i x-A_\theta(P_i x)\|_2^2
+
\lambda_{TV}\mathrm{TV}(x)
\]

其中：

\[
D(x)=\frac{1}{2}\|M\odot x-y\|_2^2
\]

\[
R_{AE}(x)=\sum_i\|P_i x-A_\theta(P_i x)\|_2^2
\]

---

## 18. TV loss

推荐 PyTorch 版本：

```python
def tv_loss_torch(x, eps=1e-6):
    dx = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]
    dy = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]
    return torch.sqrt(dx ** 2 + dy ** 2 + eps).mean()
```

---

## 19. 完整优化函数框架

```python
def optimize_with_ae_tv(
    y,
    mask,
    ae,
    patch_size=8,
    stride=4,
    lam_ae=1e-2,
    lam_tv=1e-3,
    lr=1e-2,
    steps=1000,
    init_x=None,
):
    H, W = y.shape

    y_t = torch.tensor(y, dtype=torch.float32).view(1, 1, H, W)
    m_t = torch.tensor(mask, dtype=torch.float32).view(1, 1, H, W)

    if init_x is None:
        x0 = y.copy()
    else:
        x0 = init_x.copy()

    x = torch.tensor(x0, dtype=torch.float32).view(1, 1, H, W)
    x.requires_grad_(True)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam([x], lr=lr)

    log = {
        "total": [],
        "data": [],
        "ae": [],
        "tv": [],
    }

    for step in range(steps):
        optimizer.zero_grad()

        data_loss = 0.5 * ((m_t * x - y_t) ** 2).mean()
        ae_loss = ae_prior_loss_torch(
            x,
            ae,
            patch_size=patch_size,
            stride=stride,
            reduction="mean"
        )
        tv = tv_loss_torch(x)

        total_loss = data_loss + lam_ae * ae_loss + lam_tv * tv

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)

        log["total"].append(float(total_loss.item()))
        log["data"].append(float(data_loss.item()))
        log["ae"].append(float(ae_loss.item()))
        log["tv"].append(float(tv.item()))

    x_hat = x.detach().cpu().numpy()[0, 0]
    return x_hat, log
```

---

## 20. Step 9：跑 Ours 的最小闭环

### 20.1 目标

确认：

- AE 权重可以加载；
- Ours 能跑完；
- loss 能下降；
- 输出图像正常；
- PSNR/SSIM 正常；
- Ours 至少不要明显差于 TV。

### 20.2 建议设置

```text
image = cameraman
mask = random30
AE weight = cameraman_p8_d16_current.pt
methods = TV, K-means, Ours
```

### 20.3 输出路径

```text
results/debug_optimization/
```

### 20.4 输出文件

```text
cameraman_random30_original.png
cameraman_random30_corrupted.png
cameraman_random30_tv.png
cameraman_random30_kmeans.png
cameraman_random30_ours.png
cameraman_random30_ours_log.npy
```

---

# Part IV：参数调试阶段

---

## 21. Step 10：调 Ours 的关键参数

### 21.1 默认设置

建议先固定：

```text
patch_size = 8
latent_dim = 16
stride = 4
steps = 1000
lr = 0.01
lambda_TV = 0.001
```

### 21.2 重点调 \(\lambda_{AE}\)

测试：

\[
\lambda_{AE}
\in
\{10^{-4},10^{-3},10^{-2},5\times10^{-2},10^{-1}\}
\]

### 21.3 观察现象

| 情况 | 可能现象 |
|---|---|
| \(\lambda_{AE}\) 太小 | Ours 接近 TV，AE prior 没明显作用 |
| \(\lambda_{AE}\) 太大 | 可能产生 artifacts |
| 适中 | 纹理和结构较好，结果更平衡 |

### 21.4 推荐初始值

可以先用：

```text
lambda_AE = 1e-2
lambda_TV = 1e-3
```

---

## 22. Step 11：判断现有 AE 权重是否可用于正式实验

你现在的权重命名是：

```text
cameraman_p8_d16_current.pt
barbara_p8_d16_current.pt
...
```

它们没有区分 mask 类型。

### 22.1 调试阶段

可以直接用这些权重调试 pipeline。

### 22.2 正式实验阶段

更推荐按 image + mask type 重新训练或保存：

```text
cameraman_random30_p8_d16_current.pt
cameraman_random50_p8_d16_current.pt
cameraman_random70_p8_d16_current.pt
cameraman_block_p8_d16_current.pt
cameraman_irregular_p8_d16_current.pt
```

原因：

你们的方法是从 corrupted image 的 visible patches 学习 image-specific prior。不同 mask 的 visible patches 不完全一样，因此理论上 AE 权重也应该区分 mask。

### 22.3 需要向 A 确认的问题

你要问 A：

```text
这些 AE 权重训练时传入的是 original image 还是 corrupted image？
如果 visible_ratio_thresh 小于 1，是否可能用到了 missing pixels 的 ground truth？
```

如果训练时用了 original image 且 threshold 小于 1，可能存在 ground-truth leakage。

---

# Part V：主实验阶段

---

## 23. Step 12：跑主实验

### 23.1 图像

先用 5 张：

```text
cameraman
barbara
house
peppers
lena
```

之后有时间再加 BSD68 子集。

### 23.2 Mask 类型

跑：

```text
random30
random50
random70
block
irregular
```

### 23.3 方法

跑：

```text
OpenCV Telea
OpenCV Navier-Stokes
TV inpainting
Wavelet sparse inpainting
K-means patch prior
Ours
```

### 23.4 总实验量

\[
5 \text{ images}
\times
5 \text{ masks}
\times
6 \text{ methods}
=
150
\]

组结果。

---

## 24. Step 13：保存主实验结果

### 24.1 恢复图像保存路径

```text
results/restored_images/
```

命名方式：

```text
{image}_{mask_type}_{method}.png
```

例如：

```text
barbara_random50_ours.png
barbara_random50_tv.png
barbara_random50_kmeans.png
```

### 24.2 日志保存路径

```text
results/logs/
```

例如：

```text
barbara_random50_ours_log.npy
```

### 24.3 表格保存路径

```text
results/tables/main_results.csv
```

CSV 每一行：

```text
image, mask_type, method, psnr, ssim, rmse, mae, runtime
```

例如：

```text
barbara, random50, tv, 25.31, 0.721, 0.054, 0.031, 8.2
barbara, random50, ours, 27.08, 0.779, 0.044, 0.024, 16.5
```

---

# Part VI：B 负责的消融实验阶段

---

## 25. Step 14：Objective term ablation

这是你作为 B 最重要的消融实验。

### 25.1 比较模型

#### 1. TV only

\[
D(x)+\lambda_{TV}\mathrm{TV}(x)
\]

#### 2. AE only

\[
D(x)+\lambda_{AE}R_{AE}(x)
\]

#### 3. AE + TV

\[
D(x)+\lambda_{AE}R_{AE}(x)+\lambda_{TV}\mathrm{TV}(x)
\]

### 25.2 目的

证明：

- TV prior 负责平滑和稳定；
- AE prior 负责 patch-level structure；
- AE + TV 最平衡。

### 25.3 输出表格

```text
results/tables/objective_ablation.csv
```

表格格式：

| Model | PSNR ↑ | SSIM ↑ | RMSE ↓ | Time ↓ |
|---|---:|---:|---:|---:|
| TV only | | | | |
| AE only | | | | |
| AE + TV | | | | |

---

## 26. Step 15：\(\lambda_{AE}\) sensitivity

### 26.1 测试参数

\[
\lambda_{AE}
\in
\{10^{-4},10^{-3},10^{-2},5\times10^{-2},10^{-1}\}
\]

### 26.2 固定参数

```text
lambda_TV = 1e-3
patch_size = 8
latent_dim = 16
steps = 1000
```

### 26.3 输出

```text
results/tables/lambda_ae_sensitivity.csv
```

表格格式：

| \(\lambda_{AE}\) | PSNR ↑ | SSIM ↑ | RMSE ↓ |
|---:|---:|---:|---:|
| 1e-4 | | | |
| 1e-3 | | | |
| 1e-2 | | | |
| 5e-2 | | | |
| 1e-1 | | | |

---

## 27. Step 16：Convergence curve

### 27.1 推荐图像

```text
barbara + random50
```

### 27.2 记录内容

每隔 20 或 50 step 记录：

```text
total loss
data loss
AE loss
TV loss
PSNR
SSIM
```

### 27.3 输出

日志：

```text
results/logs/barbara_random50_ours_log.npy
```

图像：

```text
results/figures/convergence_loss.png
results/figures/convergence_psnr.png
```

### 27.4 论文中说明

这组实验用来证明：

> The proposed optimization process is stable, and the restoration quality gradually improves or stabilizes during iterations.

---

# Part VII：图表整理阶段

---

## 28. Step 17：生成主结果表 Table 1

### 28.1 内容

对所有图像和 mask 求平均。

| Method | Type | PSNR ↑ | SSIM ↑ | RMSE ↓ | Time ↓ |
|---|---|---:|---:|---:|---:|
| OpenCV Telea | Classical | | | | |
| OpenCV Navier-Stokes | PDE-inspired | | | | |
| TV Inpainting | Optimization | | | | |
| Wavelet Sparse | Optimization | | | | |
| K-means Patch Prior | Patch prior | | | | |
| Ours | AE-prior optimization | | | | |

---

## 29. Step 18：生成不同 mask 下的结果 Table 2

### 29.1 内容

建议先只放 PSNR。

| Method | Random30 | Random50 | Random70 | Block | Irregular |
|---|---:|---:|---:|---:|---:|
| OpenCV Telea | | | | | |
| OpenCV NS | | | | | |
| TV | | | | | |
| Wavelet | | | | | |
| K-means | | | | | |
| Ours | | | | | |

---

## 30. Step 19：生成 qualitative comparison figure

### 30.1 推荐选择

选三组：

```text
Barbara + random50
Cameraman + block
Peppers + irregular
```

### 30.2 每行展示

```text
Original | Masked | Telea | NS | TV | Wavelet | K-means | Ours
```

如果太长，可以分成两张图。

---

## 31. Step 20：生成 convergence figure

### 31.1 图 1

```text
iteration vs total loss
```

也可以同时画：

```text
data loss
AE loss
TV loss
```

### 31.2 图 2

```text
iteration vs PSNR
```

---
