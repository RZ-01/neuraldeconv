import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

# 必须在 import cv2 之前设置
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "2000000000"

import cv2
import numpy as np
from scipy.ndimage import convolve


# =============================================================================
# 图像读取（不变）
# =============================================================================

def normalize_image(img: np.ndarray) -> np.ndarray:
    """将 uint8、uint16 或浮点图像归一化到 float32 [0, 1]。"""
    original_dtype = img.dtype
    img = img.astype(np.float32)

    if np.issubdtype(original_dtype, np.integer):
        max_value = float(np.iinfo(original_dtype).max)
        img /= max_value
    elif img.size > 0 and img.max() > 1.0:
        if img.max() <= 255.0:
            img /= 255.0
        elif img.max() <= 65535.0:
            img /= 65535.0
        else:
            img /= float(img.max())

    return np.clip(img, 0.0, 1.0)


def load_input_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"无法读取输入图像：{path}")

    if img.ndim == 2:
        return normalize_image(img)

    if img.ndim == 3:
        channels = img.shape[2]

        if channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif channels == 1:
            img = img[:, :, 0]
        elif channels != 3:
            raise ValueError(f"不支持的输入通道数量：shape={img.shape}")

        return normalize_image(img)

    raise ValueError(f"不支持的输入图像形状：{img.shape}")


def load_psf(path: str) -> np.ndarray:
    psf = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if psf is None:
        raise ValueError(f"无法读取 PSF：{path}")

    if psf.ndim == 3:
        if psf.shape[2] == 4:
            psf = cv2.cvtColor(psf, cv2.COLOR_BGRA2GRAY)
        elif psf.shape[2] == 3:
            psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
        elif psf.shape[2] == 1:
            psf = psf[:, :, 0]
        else:
            raise ValueError(f"不支持的 PSF 通道数量：shape={psf.shape}")

    if psf.ndim != 2:
        raise ValueError(f"PSF 必须是二维图像：shape={psf.shape}")

    psf = normalize_image(psf)
    psf_sum = float(psf.sum())

    if not np.isfinite(psf_sum) or psf_sum <= 0:
        raise ValueError(f"PSF 的总和必须大于 0，当前 sum={psf_sum}")

    psf /= psf_sum
    return psf.astype(np.float32)


# =============================================================================
# CPU Richardson-Lucy，reflect 边界，无需 padding
# =============================================================================

def rl_reflect(
    image: np.ndarray,
    psf: np.ndarray,
    psf_flipped: np.ndarray,
    num_iter: int,
    filter_epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Richardson-Lucy，内部卷积全部使用 reflect 边界。
    
    - 前向：convolve(u, psf,         mode='reflect')
    - 后向：convolve(r, psf_flipped, mode='reflect')  ← 转置算子
    
    边界处不存在"看到零"的问题，伪影从根源上消除。
    """
    u = np.full_like(image, image.mean(), dtype=np.float32)

    for _ in range(num_iter):
        conv = convolve(u, psf, mode="reflect")
        ratio = np.where(conv < filter_epsilon, 0.0, image / conv)
        u *= convolve(ratio, psf_flipped, mode="reflect")

    return u


def deconvolve_single_channel(
    channel: np.ndarray,
    psf: np.ndarray,
    psf_flipped: np.ndarray,
    num_iter: int,
) -> np.ndarray:
    channel = np.maximum(channel, 1e-10).astype(np.float32)
    return rl_reflect(channel, psf, psf_flipped, num_iter)


def run_rl(
    image: np.ndarray,
    psf: np.ndarray,
    num_iter: int,
) -> np.ndarray:
    psf_flipped = np.flip(psf).copy()  # copy 保证内存连续

    if image.ndim == 2:
        print("处理灰度图像……")
        result = deconvolve_single_channel(image, psf, psf_flipped, num_iter)

    else:
        channel_count = image.shape[2]
        print(f"处理彩色图像，共 {channel_count} 个通道（并行）……")

        def _process(cid: int) -> tuple[int, np.ndarray]:
            return cid, deconvolve_single_channel(
                image[:, :, cid], psf, psf_flipped, num_iter
            )

        channel_results = [None] * channel_count

        # scipy.ndimage 会释放 GIL，ThreadPoolExecutor 可以真正并行
        with ThreadPoolExecutor(max_workers=channel_count) as executor:
            for cid, ch_result in executor.map(_process, range(channel_count)):
                channel_results[cid] = ch_result
                print(f"  通道 {cid + 1}/{channel_count} 完成")

        result = np.stack(channel_results, axis=-1)

    finite_mask = np.isfinite(result)

    if not finite_mask.all():
        invalid_count = int((~finite_mask).sum())
        invalid_ratio = invalid_count / result.size
        raise FloatingPointError(
            f"RL 数值发散：检测到 {invalid_count} 个 NaN/Inf，"
            f"占全部像素的 {invalid_ratio:.6%}"
        )

    below_zero_ratio = float(np.mean(result < 0.0))
    above_one_ratio = float(np.mean(result > 1.0))

    print(f"  Before clipping:")
    print(f"    min={result.min():.6e}")
    print(f"    max={result.max():.6e}")
    print(f"    mean={result.mean():.6e}")
    print(f"    below 0 ratio={below_zero_ratio:.6%}")
    print(f"    above 1 ratio={above_one_ratio:.6%}")

    return np.clip(result, 0.0, 1.0)


# =============================================================================
# Main（不变）
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    type=str, required=True)
    parser.add_argument("--psf",      type=str, required=True)
    parser.add_argument("--out_dir",  type=str, required=True)
    parser.add_argument("--num_iter", type=int, default=100)
    parser.add_argument("--flip_psf", action="store_true")
    args = parser.parse_args()

    if args.num_iter <= 0:
        raise ValueError(f"--num_iter 必须大于 0，当前值：{args.num_iter}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"读取输入图像：{args.input}")
    image = load_input_image(args.input)
    height, width = image.shape[:2]

    if image.ndim == 2:
        print(f"  Shape: {height} x {width}, grayscale")
    else:
        print(f"  Shape: {height} x {width}, channels={image.shape[2]}")

    print(f"  Range: [{image.min():.6f}, {image.max():.6f}]")

    print(f"读取 PSF：{args.psf}")
    psf = load_psf(args.psf)
    print(f"  PSF shape: {psf.shape}")
    print(f"  PSF sum: {psf.sum():.8f}")

    if args.flip_psf:
        psf = np.flip(psf, axis=(0, 1)).copy()
        print("  PSF 已翻转")
    else:
        print("  PSF 未翻转")

    print(f"\n开始 Richardson-Lucy，迭代次数：{args.num_iter}")

    start_time = time.time()
    deconvolved = run_rl(image=image, psf=psf, num_iter=args.num_iter)
    elapsed = time.time() - start_time

    print(f"\nRL 完成，耗时：{elapsed:.2f} 秒")
    print(f"输出 shape：{deconvolved.shape}")
    print(f"输出范围：[{deconvolved.min():.6f}, {deconvolved.max():.6f}]")

    output_path = os.path.join(args.out_dir, "rl_deconvolved.png")
    output_uint8 = np.round(deconvolved * 255.0).astype(np.uint8)
    success = cv2.imwrite(output_path, output_uint8)

    if not success:
        raise IOError(f"无法保存输出图像：{output_path}")

    print(f"保存成功：{output_path}")


if __name__ == "__main__":
    main()