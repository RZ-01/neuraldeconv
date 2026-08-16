import argparse
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "2000000000"
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.utils.checkpoint

from models.instantngp import InstantNGPTorchModel

class ImageDataset(Dataset):
    def __init__(self, norm_image_tensor, num_pixels_per_step, num_batches):
        self.norm_image_tensor = norm_image_tensor
        self.num_pixels_per_step = num_pixels_per_step
        self.num_batches = num_batches
        self.image_shape = norm_image_tensor.shape

        if len(self.image_shape) != 2:
            raise ValueError(f"Only 2D images are supported, got shape={self.image_shape}")

        h, w = self.image_shape
        self.inv_shape = torch.tensor(
            [1.0 / max(h - 1, 1), 1.0 / max(w - 1, 1)],
            dtype=torch.float32,
        )

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        h, w = self.image_shape
        y_idx_t = torch.randint(0, h, (self.num_pixels_per_step,))
        x_idx_t = torch.randint(0, w, (self.num_pixels_per_step,))
        target_coords = torch.stack([y_idx_t, x_idx_t], dim=1)

        y_idx = target_coords[:, 0].numpy()
        x_idx = target_coords[:, 1].numpy()
        values = self.norm_image_tensor[y_idx, x_idx]
        target_values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        target_coords_normalized = target_coords.float() * self.inv_shape

        return {
            'target_coords': target_coords_normalized,   # (P, 2)
            'target_values': target_values,              # (P,)
        }


def generate_offsets_on_gpu(
    n: int,
    discrete_psf: torch.Tensor,
) -> torch.Tensor:
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


def reflect_coords(source_coords: torch.Tensor) -> torch.Tensor:
    s = source_coords.abs()
    return (1.0 - (s - 1.0).abs()).clamp(0.0, 1.0)


def psf_uniform_sampling_step(
    model: nn.Module,
    batch: dict,
    num_mc_samples: int,
    device: torch.device,
    discrete_psf: torch.Tensor,
    inv_shape: torch.Tensor,
    stochastic_alpha: float = 0.0,
    use_reflect: bool = True,
    mc_chunk_size: int = 0,
) -> dict:
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)

    num_pixels = target_coords.shape[0]
    n_dims = target_coords.shape[1]
    inv_shape_gpu = inv_shape.to(device, non_blocking=True)

    use_chunked = 0 < mc_chunk_size < num_mc_samples

    if not use_chunked:
        sampling_budget = num_pixels * num_mc_samples
        sampled_offsets = generate_offsets_on_gpu(
            sampling_budget, discrete_psf=discrete_psf,
        )
        sampled_offsets = (sampled_offsets * inv_shape_gpu).view(num_pixels, num_mc_samples, n_dims)
        source_coords = target_coords.unsqueeze(1) + sampled_offsets
        if use_reflect:
            source_coords = reflect_coords(source_coords)
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
                offsets = generate_offsets_on_gpu(
                    num_pixels * _cs, discrete_psf=discrete_psf,
                )
                offsets = (offsets * inv_shape_gpu).view(num_pixels, _cs, n_dims)
                src = tc.unsqueeze(1) + offsets
                if use_reflect:
                    src = reflect_coords(src)
                else:
                    src = torch.clamp(src, 0.0, 1.0)
                src_flat = src.view(-1, n_dims)
                coords = torch.stack([src_flat[:, 1], src_flat[:, 0]], dim=-1).float()
                pred, _ = model(coords, variance=None, stochastic_alpha=stochastic_alpha)
                return pred.view(num_pixels, _cs).sum(dim=1)

            chunk_sum = torch.utils.checkpoint.checkpoint(
                _chunk_fn, target_coords, use_reentrant=False,
            )
            running_sum = running_sum + chunk_sum

        simulated_values = running_sum / total_samples

    data_loss = F.mse_loss(simulated_values.float(), target_values.float()) * 100

    return {
        "reconstruction_loss": data_loss,
        "total_loss":          data_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="../city_blur.png")
    parser.add_argument("--psf_path", type=str, default="../psf.png",
                        help="Path to discrete 2D PSF file (image or .npy).")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/city_chunkedMC.pth")
    parser.add_argument("--logdir", type=str, default="../runs/city_chunkedMC")    parser.add_argument("--num_mc_samples", type=int, default=4000)
    parser.add_argument("--boundary", choices=("reflect", "clamp"), default="reflect",
                        help="Boundary handling for sampled source coordinates.")
    parser.add_argument("--progressive_steps", type=int, default=300)
    parser.add_argument("--num_pixels_per_step", type=int, default=6500,
                        help="Number of pixels sampled per training step (default: 4096).")
    parser.add_argument("--mc_chunk_size", type=int, default=0,
                        help="MC samples per chunk (0=no chunking). Reduces peak GPU memory "
                             "from O(P*M) to O(P*chunk) via gradient checkpointing (~2x compute).")
    parser.add_argument("--mode", choices=("mc", "sliding_window"), default="mc",
                        help="Training mode: 'mc' (random pixels + MC) or 'sliding_window' (dense patch + MC).")
    parser.add_argument("--patch_size", type=int, default=2048,
                        help="Patch side length for sliding_window mode.")

    # Stochastic preconditioning
    parser.add_argument("--sp_alpha_init", type=float, default=0.03)
    parser.add_argument("--sp_decay_fraction", type=float, default=0.33)

    # Encoder config
    parser.add_argument("--num_levels", type=int, default=21)
    parser.add_argument("--level_dim", type=int, default=2)
    parser.add_argument("--base_resolution", type=int, default=16)
    parser.add_argument("--log2_hashmap_size", type=int, default=24)
    parser.add_argument("--desired_resolution", type=int, default=1600)

    # Decoder config
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {args.image_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"Only 2D images are supported, got shape={image.shape}")

    if np.issubdtype(image.dtype, np.integer):
        image_norm = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image_norm = image.astype(np.float32)
        max_val = float(np.max(image_norm)) if image_norm.size > 0 else 1.0
        if max_val > 1.0:
            image_norm = image_norm / max_val

    print(f"Image loaded: shape={image_norm.shape}")
    num_pixels_per_step = args.num_pixels_per_step
    n_dims = 2
    # calculate mc sample count and steps based on input image
    # Image (H, W) PSF size(h,w)
    # pixel per step: 6500
    # steps: HxWx160/6500
    # mc samples: hxw x 5.4869
    h, w = image_norm.shape
    num_pixels_per_step = 6500
    steps = int(h * w * 160 / num_pixels_per_step)

    encoder_config = {
        "otype": "HashGrid",
        "n_levels": args.num_levels,
        "n_features_per_level": args.level_dim,
        "log2_hashmap_size": args.log2_hashmap_size,
        "base_resolution": args.base_resolution,
        "per_level_scale": np.exp(
            (np.log(args.desired_resolution) - np.log(args.base_resolution)) / (args.num_levels - 1)
        ),
    }
    decoder_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": args.hidden_dim,
        "n_hidden_layers": args.num_layers,
    }

    model = InstantNGPTorchModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_input_dims=n_dims,
        learn_variance=False,
    ).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Load discrete PSF ─────────────────────────────────────────────────
    if args.psf_path.endswith(".npy"):
        discrete_psf_np = np.load(args.psf_path).astype(np.float32)
    else:
        discrete_psf_np = cv2.imread(args.psf_path, cv2.IMREAD_UNCHANGED)
        if discrete_psf_np is None:
            raise ValueError(f"Unreadable PSF: {args.psf_path}")
        if discrete_psf_np.ndim == 3:
            discrete_psf_np = cv2.cvtColor(discrete_psf_np, cv2.COLOR_BGR2GRAY)
        discrete_psf_np = discrete_psf_np.astype(np.float32)
    if discrete_psf_np.ndim != 2:
        raise ValueError(f"PSF must be 2D, got shape={discrete_psf_np.shape}")
    psf_sum = float(discrete_psf_np.sum())
    if psf_sum <= 0:
        raise ValueError("PSF sum must be positive.")
    discrete_psf_np /= psf_sum
    discrete_psf = torch.from_numpy(discrete_psf_np).float().to(device)
    # scipy.ndimage.convolve flips the kernel; flip here so the forward model matches
    # discrete_psf = torch.flip(discrete_psf, [0, 1])
    print(f"PSF loaded: shape={discrete_psf.shape}")
    h_psf, w_psf = discrete_psf.shape
    num_mc_samples = int(h_psf * w_psf * 4000 / 729)  # scale with PSF size, but max 4000
    print(f"Calculated steps: {steps}, num_mc_samples: {num_mc_samples}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=args.logdir)

    if args.mode == "mc":
        dataset = ImageDataset(
            norm_image_tensor=image_norm,
            num_pixels_per_step=num_pixels_per_step,
            num_batches=steps,
        )
        inv_shape = dataset.inv_shape
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        data_iter = iter(dataloader)
    else:
        inv_shape = torch.tensor(
            [1.0 / max(h - 1, 1), 1.0 / max(w - 1, 1)],
            dtype=torch.float32,
        )

    n_levels = args.num_levels
    initial_levels = 4
    steps_per_level = args.progressive_steps // max(n_levels - initial_levels, 1)
    sp_decay_steps = int(steps * args.sp_decay_fraction)
    last_set_level = -1

    if args.mode == "sliding_window":
        from sliding_window import sliding_window_step

    pbar = tqdm(total=steps, desc="Training", dynamic_ncols=True)

    for step in range(steps):
        current_level = min(initial_levels + (step // steps_per_level), n_levels)
        if current_level != last_set_level:
            model.set_max_level(current_level)
            last_set_level = current_level

        optimizer.zero_grad(set_to_none=True)

        if step < sp_decay_steps:
            progress = step / sp_decay_steps
            current_alpha = args.sp_alpha_init * np.exp(-5.0 * progress)
        else:
            current_alpha = 0.0

        if args.mode == "mc":
            batch = next(data_iter)
            batch = {k: (v.squeeze(0) if hasattr(v, 'squeeze') else v) for k, v in batch.items()}
            loss_dict = psf_uniform_sampling_step(
                model=model,
                batch=batch,
                num_mc_samples=num_mc_samples,
                device=device,
                discrete_psf=discrete_psf,
                inv_shape=inv_shape,
                stochastic_alpha=current_alpha,
                use_reflect=(args.boundary == "reflect"),
                mc_chunk_size=args.mc_chunk_size,
            )
        else:
            loss_dict = sliding_window_step(
                model=model,
                image_norm=image_norm,
                patch_size=args.patch_size,
                num_mc_samples=num_mc_samples,
                device=device,
                discrete_psf=discrete_psf,
                inv_shape=inv_shape,
                stochastic_alpha=current_alpha,
                use_reflect=(args.boundary == "reflect"),
                mc_chunk_size=args.mc_chunk_size,
            )

        for key, value in loss_dict.items():
            writer.add_scalar(f"train/{key}", value.item(), step)

        loss = loss_dict["total_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        scaler.step(optimizer)
        scaler.update()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        writer.add_scalar("train/LearningRate", current_lr, step)
        writer.add_scalar("train/GradNorm", grad_norm.item(), step)

        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss_dict["reconstruction_loss"].item():.6f}'})

    pbar.close()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
    }, args.save_path)
    print(f"\nSaved model to {args.save_path}")

    writer.close()


if __name__ == "__main__":
    main()