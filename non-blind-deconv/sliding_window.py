import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np


def _generate_offsets_on_gpu(n, discrete_psf):
    psf_flat = discrete_psf.flatten()
    psf_flat = psf_flat / psf_flat.sum()
    idx = torch.multinomial(psf_flat, n, replacement=True)
    h, w = discrete_psf.shape
    y_idx = idx // w
    x_idx = idx % w
    return torch.stack([
        y_idx.float() - (h - 1) / 2.0,
        x_idx.float() - (w - 1) / 2.0,
    ], dim=1)


def _reflect_coords(source_coords):
    s = source_coords.abs()
    return (1.0 - (s - 1.0).abs()).clamp(0.0, 1.0)


def sliding_window_step(
    model: nn.Module,
    image_norm: np.ndarray,
    patch_size: int,
    num_mc_samples: int,
    device: torch.device,
    discrete_psf: torch.Tensor,
    inv_shape: torch.Tensor,
    stochastic_alpha: float = 0.0,
    use_reflect: bool = True,
    mc_chunk_size: int = 0,
    crop_region: tuple = None,
) -> dict:
    h, w = image_norm.shape

    if crop_region is not None:
        cy1, cy2, cx1, cx2 = crop_region
        cy1, cy2 = max(cy1, 0), min(cy2, h)
        cx1, cx2 = max(cx1, 0), min(cx2, w)
    else:
        cy1, cy2, cx1, cx2 = 0, h, 0, w

    region_h = cy2 - cy1
    region_w = cx2 - cx1
    ph = min(patch_size, region_h)
    pw = min(patch_size, region_w)

    y0 = cy1 + torch.randint(0, max(region_h - ph + 1, 1), (1,)).item()
    x0 = cx1 + torch.randint(0, max(region_w - pw + 1, 1), (1,)).item()

    inv_shape_gpu = inv_shape.to(device, non_blocking=True)
    yy = torch.arange(y0, y0 + ph, dtype=torch.float32, device=device)
    xx = torch.arange(x0, x0 + pw, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    target_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)
    target_coords_normalized = target_coords * inv_shape_gpu

    patch = image_norm[y0:y0 + ph, x0:x0 + pw]
    target_values = torch.from_numpy(
        np.ascontiguousarray(patch.reshape(-1), dtype=np.float32)
    ).to(device, non_blocking=True)

    num_pixels = target_coords_normalized.shape[0]
    n_dims = 2
    use_chunked = 0 < mc_chunk_size < num_mc_samples

    if not use_chunked:
        sampling_budget = num_pixels * num_mc_samples
        sampled_offsets = _generate_offsets_on_gpu(sampling_budget, discrete_psf=discrete_psf)
        sampled_offsets = (sampled_offsets * inv_shape_gpu).view(num_pixels, num_mc_samples, n_dims)
        source_coords = target_coords_normalized.unsqueeze(1) + sampled_offsets
        if use_reflect:
            source_coords = _reflect_coords(source_coords)
        else:
            source_coords = torch.clamp(source_coords, 0.0, 1.0)
        source_coords_flat = source_coords.view(-1, n_dims)
        coords_for_model = torch.stack([
            source_coords_flat[:, 1], source_coords_flat[:, 0],
        ], dim=-1).float()
        coords_for_model.requires_grad_(False)
        pred_flat, _ = model(coords_for_model, variance=None, stochastic_alpha=stochastic_alpha)
        simulated_values = pred_flat.view(num_pixels, num_mc_samples).mean(dim=1)
    else:
        running_sum = torch.zeros(num_pixels, device=device)
        total_samples = 0
        for c_start in range(0, num_mc_samples, mc_chunk_size):
            c_size = min(mc_chunk_size, num_mc_samples - c_start)
            total_samples += c_size

            def _chunk_fn(tc, _cs=c_size):
                offsets = _generate_offsets_on_gpu(num_pixels * _cs, discrete_psf=discrete_psf)
                offsets = (offsets * inv_shape_gpu).view(num_pixels, _cs, n_dims)
                src = tc.unsqueeze(1) + offsets
                if use_reflect:
                    src = _reflect_coords(src)
                else:
                    src = torch.clamp(src, 0.0, 1.0)
                src_flat = src.view(-1, n_dims)
                coords = torch.stack([src_flat[:, 1], src_flat[:, 0]], dim=-1).float()
                pred, _ = model(coords, variance=None, stochastic_alpha=stochastic_alpha)
                return pred.view(num_pixels, _cs).sum(dim=1)

            chunk_sum = torch.utils.checkpoint.checkpoint(
                _chunk_fn, target_coords_normalized, use_reentrant=False,
            )
            running_sum = running_sum + chunk_sum

        simulated_values = running_sum / total_samples

    data_loss = F.mse_loss(simulated_values.float(), target_values.float()) * 100

    return {
        "reconstruction_loss": data_loss,
        "total_loss": data_loss,
    }
