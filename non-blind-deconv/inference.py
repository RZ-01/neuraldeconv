"""
----------------
Infer the INR-fitted image ("Ours"), compare against GT, RL, and Deblur-INR.

Outputs
-------
  <out_dir>/
    inferred.png            – network prediction (uint8)
    comparison.pdf          – row0: [GT | RL | Deblur-INR | Ours]
                               row1: [blank | error(RL) | error(Deblur-INR) | error(Ours)]
    metrics.txt             – PSNR/SSIM/MSE numbers

PSNR is computed in float32 [0, 1] space.
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.instantngp import InstantNGPTorchModel


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_gray_norm(path: str) -> np.ndarray:
    """Load an image as float32 grayscale normalised to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0          # assume uint8 range
    return img

"""
def align_to_reference(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    Align src to ref via phase correlation (sub-pixel translation only).
    src_f32 = (src * 255).astype(np.float32)
    ref_f32 = (ref * 255).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(src_f32, ref_f32)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    H, W = src.shape
    shifted = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return shifted, (dx, dy)
"""

def align_to_reference(
    src: np.ndarray,
    ref: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    检查四种情况：
      1. 不对齐
      2. phase correlation 亚像素对齐
      3. phase correlation 整数像素对齐
      4. 反方向亚像素对齐

    暂时选择 MSE 最低的结果，用于诊断对齐问题。
    """
    src = np.asarray(src, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)

    if src.shape != ref.shape:
        raise ValueError(
            f"Alignment shape mismatch: src={src.shape}, ref={ref.shape}"
        )

    H, W = src.shape

    # 去除 DC 分量，避免大面积白色背景影响相关峰
    src_phase = src - src.mean()
    ref_phase = ref - ref.mean()

    # 减少 FFT 周期边界的影响
    window = cv2.createHanningWindow(
        (W, H),
        cv2.CV_32F,
    )

    (dx, dy), response = cv2.phaseCorrelate(
        src_phase,
        ref_phase,
        window,
    )

    print(
        f"    Phase shift=({dx:.4f}, {dy:.4f}), "
        f"response={response:.6f}"
    )

    def warp(
        shift_x: float,
        shift_y: float,
        interpolation: int,
    ) -> np.ndarray:
        transform = np.float32([
            [1.0, 0.0, shift_x],
            [0.0, 1.0, shift_y],
        ])

        return cv2.warpAffine(
            src,
            transform,
            (W, H),
            flags=interpolation,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)

    dx_integer = int(round(dx))
    dy_integer = int(round(dy))

    candidates = [
        ("unaligned", src, 0.0, 0.0),
        (
            "phase-subpixel",
            warp(dx, dy, cv2.INTER_LINEAR),
            dx,
            dy,
        ),
        (
            "phase-integer",
            warp(dx_integer, dy_integer, cv2.INTER_NEAREST),
            float(dx_integer),
            float(dy_integer),
        ),
        (
            "phase-reverse",
            warp(-dx, -dy, cv2.INTER_LINEAR),
            -dx,
            -dy,
        ),
    ]

    scored_candidates = []

    for label, image, shift_x, shift_y in candidates:
        image = np.clip(image, 0.0, 1.0)

        mse = float(np.mean(
            (image.astype(np.float64) - ref.astype(np.float64)) ** 2
        ))

        psnr = (
            float("inf")
            if mse == 0
            else 10.0 * np.log10(1.0 / mse)
        )

        print(
            f"    {label:16s}: "
            f"shift=({shift_x:.4f}, {shift_y:.4f}), "
            f"PSNR={psnr:.4f} dB, MSE={mse:.6e}"
        )

        scored_candidates.append(
            (mse, label, image, shift_x, shift_y)
        )

    best_mse, best_label, best_image, best_dx, best_dy = min(
        scored_candidates,
        key=lambda item: item[0],
    )

    print(
        f"    Selected alignment: {best_label}, "
        f"shift=({best_dx:.4f}, {best_dy:.4f})"
    )

    return best_image, (best_dx, best_dy)

def build_2d_coords(height: int, width: int, device: torch.device,
                    pad_h: int = 0, pad_w: int = 0) -> torch.Tensor:
    H_full = height + 2 * pad_h
    W_full = width  + 2 * pad_w
    ys = torch.arange(height, device=device, dtype=torch.float32) + pad_h
    xs = torch.arange(width,  device=device, dtype=torch.float32) + pad_w
    ys_n = ys / max(H_full - 1, 1)
    xs_n = xs / max(W_full - 1, 1)
    gy, gx = torch.meshgrid(ys_n, xs_n, indexing="ij")
    return torch.stack([gx, gy], dim=-1).view(-1, 2)


@torch.no_grad()
def infer_image(model: InstantNGPTorchModel, height: int, width: int,
                device: torch.device, batch_size: int = 500_000,
                pad_h: int = 0, pad_w: int = 0) -> np.ndarray:
    coords = build_2d_coords(height, width, device, pad_h=pad_h, pad_w=pad_w)
    preds = []
    for i in range(0, coords.shape[0], batch_size):
        chunk = coords[i : i + batch_size]
        pred, _ = model(chunk, variance=None)
        preds.append(pred.cpu())
    pred_flat = torch.cat(preds, dim=0).numpy()
    img = pred_flat.reshape(height, width).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def load_align_baseline(path: str, gt: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Load a baseline image, resize to GT resolution, align via phase correlation."""
    H, W = gt.shape
    raw = load_gray_norm(path)
    if raw.shape != gt.shape:
        print(f"    Resizing baseline {raw.shape} → {gt.shape}")
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
    aligned, shift = align_to_reference(raw, gt)
    return aligned, shift


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred_01 = pred.clip(0, 1)
    gt_01 = gt.clip(0, 1)
    psnr = peak_signal_noise_ratio(gt_01, pred_01, data_range=1.0)
    ssim = structural_similarity(gt_01, pred_01, data_range=1.0)
    mse  = mean_squared_error(gt_01, pred_01)
    return psnr, ssim, mse


def error_map(pred: np.ndarray, gt: np.ndarray, cmap: str = "hot",
              vmax: float | None = None) -> np.ndarray:
    """Absolute error as an RGB uint8 image. Pass vmax for a shared scale."""
    err = np.abs(pred - gt)
    print(f"  Error map stats: max={err.max():.6f}, mean={err.mean():.6f}, min={err.min():.6f}")
    scale = vmax if vmax is not None else max(float(err.max()), 1e-8)
    err_norm = np.clip(err / scale, 0.0, 1.0)
    cmap_fn = plt.get_cmap(cmap)
    rgb = (cmap_fn(err_norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def save_comparison(gt: np.ndarray, results: list[tuple[str, np.ndarray]],
                    metrics: list[tuple[float, float, float]],
                    out_path: str):
    """
    Row 0: GT | <result_1> | <result_2> | ... | <result_N>
    Row 1: (blank) | Error(result_1) | Error(result_2) | ... | Error(result_N)
    """
    def to_rgb(arr: np.ndarray) -> np.ndarray:
        return np.stack([arr, arr, arr], axis=-1)

    H, W = gt.shape
    n_cols = 1 + len(results)   # GT column + one column per result
    col_w  = 5.0
    fig_w  = col_w * n_cols
    row_h  = col_w / (W / H) + 0.3
    fig_h  = row_h * 2

    shared_vmax = max(
        max(float(np.abs(img - gt).max()) for _, img in results),
        1e-8,
    )

    fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.1})

    # Row 0, col 0: GT. Row 1, col 0: blank.
    axes[0, 0].imshow(to_rgb(gt))
    axes[0, 0].set_title("GT", fontsize=8, pad=4)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")   # left empty on purpose

    for i, ((label, img), (psnr, ssim, mse)) in enumerate(zip(results, metrics), start=1):
        axes[0, i].imshow(to_rgb(img))
        axes[0, i].set_title(f"{label}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}  MSE={mse:.2e}",
                              fontsize=8, pad=4)
        axes[0, i].axis("off")

        err_rgb = error_map(img, gt, vmax=shared_vmax)
        axes[1, i].imshow(err_rgb)
        axes[1, i].set_title(f"Error ({label} vs GT)", fontsize=8, pad=4)
        axes[1, i].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infer INR ('Ours'), compare vs GT / RL / Deblur-INR, output PSNR + visual."
    )
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--gt_image",     type=str, required=True)
    parser.add_argument("--baseline_rl_image",         type=str, default=None,
                         help="Path to Richardson-Lucy result image.")
    parser.add_argument("--baseline_deblur_inr_image", type=str, default=None,
                         help="Path to Deblur-INR result image.")
    parser.add_argument("--out_dir",      type=str, default="../inference_2d")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--batch_size",   type=int, default=500_000)
    parser.add_argument("--psf_path", type=str, default=None,
                    help="PSF path (informational only; no longer used for coord shift).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load GT ──────────────────────────────────────────────────────────────
    print(f"Loading GT image: {args.gt_image}")
    gt = load_gray_norm(args.gt_image)
    H, W = gt.shape
    print(f"  GT shape: {H} x {W}")
    gt_01 = gt.clip(0, 1)

    # Blur was generated with reflect-pad → INR domain == image domain, no coord shift needed
    pad_h, pad_w = 0, 0

    # ── Load baselines (RL, Deblur-INR) ────────────────────────────────────────
    baseline_specs = [
        ("RL",         args.baseline_rl_image),
        ("Deblur-INR", args.baseline_deblur_inr_image),
    ]
    baseline_results: list[tuple[str, np.ndarray]] = []
    for label, path in baseline_specs:
        if path is None:
            continue
        print(f"Loading baseline [{label}]: {path}")
        aligned, (dx, dy) = load_align_baseline(path, gt)
        print(f"    Aligned [{label}]: shift = ({dx:.2f}, {dy:.2f}) px")
        baseline_results.append((label, aligned.clip(0, 1)))

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder_config = ckpt.get("encoder_config")
    decoder_config = ckpt.get("decoder_config")
    if encoder_config is None:
        print("  Warning: no encoder_config in checkpoint – using defaults.")

    model = InstantNGPTorchModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_input_dims=2,
        learn_variance=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("  Model loaded.")

    # ── Infer ("Ours") ──────────────────────────────────────────────────────────
    print(f"Inferring image ({H}x{W})…")
    inferred = infer_image(model, H, W, device,
                       batch_size=args.batch_size,
                       pad_h=pad_h, pad_w=pad_w)
    print(f"  Inferred range: [{inferred.min():.4f}, {inferred.max():.4f}]")
    inferred, (dx, dy) = align_to_reference(inferred, gt)
    print(f"  Aligned inferred: shift = ({dx:.2f}, {dy:.2f}) px")

    # Save raw inferred image
    inferred_uint8 = (inferred * 255).clip(0, 255).astype(np.uint8)
    inferred_path = os.path.join(args.out_dir, "inferred.png")
    cv2.imwrite(inferred_path, inferred_uint8)
    print(f"  Saved inferred image: {inferred_path}")
    inferred_01 = inferred.clip(0, 1)

    # Order: RL, Deblur-INR, Ours
    results: list[tuple[str, np.ndarray]] = list(baseline_results) + [("Ours", inferred_01)]

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = [compute_metrics(img, gt_01) for _, img in results]
    for (label, _), (psnr, ssim, mse) in zip(results, metrics):
        print(f"{label} vs GT:  PSNR={psnr:.4f} dB  SSIM={ssim:.6f}  MSE={mse:.6e}")

    txt_path = os.path.join(args.out_dir, "metrics.txt")
    with open(txt_path, "w") as f:
        for (label, _), (psnr, ssim, mse) in zip(results, metrics):
            f.write(f"{label} vs GT : PSNR={psnr:.6f} dB  SSIM={ssim:.6f}  MSE={mse:.6e}\n")
    print(f"Saved metrics: {txt_path}")

    # ── Comparison figure ─────────────────────────────────────────────────────
    comp_path = os.path.join(args.out_dir, "comparison.pdf")
    save_comparison(gt=gt_01, results=results, metrics=metrics, out_path=comp_path)

    print("\nDone.")


if __name__ == "__main__":
    main()