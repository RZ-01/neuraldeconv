"""
验证 rl_reflect 与 skimage 参考实现在图像内部的一致性。

用法：
    python verify_rl.py --input image.png --psf psf.png --num_iter 30
"""

import argparse
import numpy as np
from skimage.restoration import richardson_lucy as skrl

from . import load_input_image, load_psf, rl_reflect


def compare(image: np.ndarray, psf: np.ndarray, num_iter: int) -> None:
    # 对彩色图只取第一个通道，省去多通道循环
    if image.ndim == 3:
        image = image[:, :, 0]

    image = np.maximum(image, 1e-10).astype(np.float32)
    psf_flipped = np.flip(psf).copy()

    # ------------------------------------------------------------------
    # 两种实现
    # ------------------------------------------------------------------
    ref  = skrl(image, psf, num_iter=num_iter).astype(np.float32)
    ours = rl_reflect(image, psf, psf_flipped, num_iter).astype(np.float32)

    # ------------------------------------------------------------------
    # 只看内部：裁掉 2× PSF 尺寸的四周
    # ------------------------------------------------------------------
    ph = psf.shape[0] * 2
    pw = psf.shape[1] * 2

    ref_crop  = ref[ph:-ph, pw:-pw]
    ours_crop = ours[ph:-ph, pw:-pw]

    diff = np.abs(ref_crop - ours_crop)

    print(f"PSF shape   : {psf.shape}")
    print(f"Image shape : {image.shape}")
    print(f"Crop margin : {ph}px (h), {pw}px (w)")
    print(f"Crop shape  : {ref_crop.shape}")
    print()
    print(f"Max abs diff  : {diff.max():.6e}")
    print(f"Mean abs diff : {diff.mean():.6e}")
    print(f"RMSE          : {np.sqrt(np.mean(diff**2)):.6e}")

    threshold = 1e-3
    ok = diff.max() < threshold
    print()
    print(f"{'[PASS]' if ok else '[FAIL]'}  max diff < {threshold}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    type=str, required=True)
    parser.add_argument("--psf",      type=str, required=True)
    parser.add_argument("--num_iter", type=int, default=30)
    args = parser.parse_args()

    image = load_input_image(args.input)
    psf   = load_psf(args.psf)

    compare(image, psf, args.num_iter)


if __name__ == "__main__":
    main()