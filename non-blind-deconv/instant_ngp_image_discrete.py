import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

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
        # INR [0,1] domain spans the blurry image directly (same size as sharp after reflect-pad blur)
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

        target_coords = torch.stack([y_idx_t, x_idx_t], dim=1).float()
        target_coords_normalized = target_coords * self.inv_shape

        y_idx = y_idx_t.numpy()
        x_idx = x_idx_t.numpy()
        values = self.norm_image_tensor[y_idx, x_idx]
        target_values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))

        return {
            'target_coords': target_coords_normalized,
            'target_values': target_values,
        }


def generate_offsets_on_gpu(
    n: int,
    discrete_psf: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample n PSF offsets on the GPU.

    Steps:
      1. Sample a histogram bin via multinomial (integer pixel offset).
      2. Add uniform jitter within that pixel: U(-0.5, +0.5) per axis.

    Returns offsets of shape (n, 2), float32, in pixel units centred at PSF origin.
    """
    psf_flat = discrete_psf.flatten()
    psf_flat = psf_flat / psf_flat.sum()
    idx = torch.multinomial(psf_flat, n, replacement=True)
    h, w = discrete_psf.shape
    y_idx = idx // w
    x_idx = idx % w
    jitter = torch.rand((n, 2), device=device) - 0.5
    return torch.stack([
        y_idx.float() - (h - 1) / 2.0 + jitter[:, 0],
        x_idx.float() - (w - 1) / 2.0 + jitter[:, 1],
    ], dim=1)

def precompute_psf_grid(
    discrete_psf: torch.Tensor,
    inv_shape: torch.Tensor,
    device: torch.device,
):
    """
    Returns:
        psf_offsets_norm: (K, 2) deterministic offsets in normalized image coords,
                          K = H_psf * W_psf
        psf_weights:      (K,)   PSF weights, already sum to 1
    """
    H, W = discrete_psf.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    offsets_pix = torch.stack([
        yy - (H - 1) / 2.0,
        xx - (W - 1) / 2.0,
    ], dim=-1).view(-1, 2)                          # (K, 2) in PSF-pixel units

    psf_offsets_norm = offsets_pix * inv_shape.to(device)   # (K, 2) in normalized coords
    psf_weights = discrete_psf.flatten().contiguous()       # (K,)
    return psf_offsets_norm, psf_weights

def psf_deterministic_step(
    model: nn.Module,
    batch: dict,
    psf_offsets_norm: torch.Tensor,   # (K, 2)
    psf_weights: torch.Tensor,        # (K,)
    device: torch.device,
    stochastic_alpha: float = 0.0,
    use_reflect: bool = True,
) -> dict:
    target_coords = batch['target_coords'].to(device, non_blocking=True)  # (P, 2)
    target_values = batch['target_values'].to(device, non_blocking=True)  # (P,)

    P = target_coords.shape[0]
    K = psf_offsets_norm.shape[0]

    # (P, 1, 2) + (1, K, 2) -> (P, K, 2)
    source_coords = target_coords.unsqueeze(1) + psf_offsets_norm.unsqueeze(0)

    # Boundary handling
    if use_reflect:
        # triangle reflect into [0, 1] (works for small offsets; safe clamp at the end)
        s = source_coords.abs()
        source_coords = 1.0 - (s - 1.0).abs()
        source_coords = source_coords.clamp(0.0, 1.0)
    else:
        source_coords = source_coords.clamp(0.0, 1.0)

    source_flat = source_coords.reshape(-1, 2)
    coords_for_model = torch.stack([source_flat[:, 1], source_flat[:, 0]], dim=-1)

    pred_flat, _ = model(coords_for_model, variance=None, stochastic_alpha=stochastic_alpha)
    pred = pred_flat.view(P, K)                            # (P, K)

    # Deterministic weighted sum = (sharp ⊛ PSF) at target pixel
    simulated_values = (pred * psf_weights.unsqueeze(0)).sum(dim=1)   # (P,)

    data_loss = F.mse_loss(simulated_values.float(), target_values.float()) * 100
    return {
        "reconstruction_loss": data_loss,
        "total_loss":          data_loss,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/workspace/Deblur-INR/datasets/lai/im05_ker04.png")
    parser.add_argument("--psf_path", type=str, default="/workspace/Deblur-INR/results/ker04_truth.png",
                        help="Path to discrete 2D PSF file (image or .npy).")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--save_path", type=str, default="../checkpoints/im05_ker04.pth")
    parser.add_argument("--logdir", type=str, default="../runs/im05_ker04")
    parser.add_argument("--num_mc_samples", type=int, default=300)
    parser.add_argument("--progressive_steps", type=int, default=300)
    parser.add_argument("--num_pixels_per_step", type=int, default=24000,
                        help="Number of pixels sampled per training step (default: 4096).")

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
    discrete_psf = torch.flip(discrete_psf, [0, 1])
    print(f"PSF loaded: shape={discrete_psf.shape}")


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0)
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=args.logdir)

    h_img, w_img = image_norm.shape
    inv_shape = torch.tensor(
        [1.0 / max(h_img - 1, 1), 1.0 / max(w_img - 1, 1)],
        dtype=torch.float32,
    )
    dataset = ImageDataset(
        norm_image_tensor=image_norm,
        num_pixels_per_step=num_pixels_per_step,
        num_batches=args.steps,
    )

    psf_offsets_norm, psf_weights = precompute_psf_grid(discrete_psf, inv_shape, device)
    print(f"PSF grid: {psf_offsets_norm.shape[0]} deterministic samples per target pixel")


    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    data_iter = iter(dataloader)

    n_levels = args.num_levels
    initial_levels = 4
    steps_per_level = args.progressive_steps // max(n_levels - initial_levels, 1)
    sp_decay_steps = int(args.steps * args.sp_decay_fraction)
    last_set_level = -1

    pbar = tqdm(total=args.steps, desc="Training", dynamic_ncols=True)

    for step in range(args.steps):
        current_level = min(initial_levels + (step // steps_per_level), n_levels)
        if current_level != last_set_level:
            model.set_max_level(current_level)
            last_set_level = current_level

        batch = next(data_iter)
        batch = {k: (v.squeeze(0) if hasattr(v, 'squeeze') else v) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        if step < sp_decay_steps:
            progress = step / sp_decay_steps
            current_alpha = args.sp_alpha_init * np.exp(-5.0 * progress)
        else:
            current_alpha = 0.0

        loss_dict = psf_deterministic_step(
            model=model,
            batch=batch,
            psf_offsets_norm=psf_offsets_norm,
            psf_weights=psf_weights,
            device=device,
            stochastic_alpha=current_alpha,
            use_reflect=True,
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
