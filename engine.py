import time
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import trange

from .operators import compute_svd_basis, stripe_neighborhood_completion
from .models import HyperZSDNet, denoise
from .losses import hyperzsd_loss
from .metrics import evaluate_pair
from .visualization import plot_losses, save_reconstruction_figure
from .utils import save_json

def run_experiment(clean_img, noisy_img, config, device):
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # Spectral projection and neighborhood completion.
    basis = compute_svd_basis(noisy_img, config.spectral_dim)
    channels, height, width = noisy_img.shape[1:]
    projected = torch.mm(basis.T, noisy_img.squeeze(0).reshape(channels, -1))
    projected = projected.reshape(config.spectral_dim, height, width)
    completed = stripe_neighborhood_completion(projected, threshold_rho=config.threshold_rho, max_search_distance=config.max_search_distance,).unsqueeze(0)

    min_vals = completed.amin(dim=(2, 3), keepdim=True)
    max_vals = completed.amax(dim=(2, 3), keepdim=True)
    normalized = (completed - min_vals) / (max_vals - min_vals + 1e-6)

    model = HyperZSDNet(normalized.shape[1], config.channel_embed).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    history = {"total": [], "residual": [], "consistency": []}

    model.train()
    for epoch in trange(config.max_epoch, desc="Training"):
        total, residual, consistency = hyperzsd_loss(model, normalized, config.tau)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            history["total"].append(float(total.detach().cpu()))
            history["residual"].append(float(residual.cpu()))
            history["consistency"].append(float(consistency.cpu()))

    plot_losses(history, output / "loss_curves.png", config.show_plots)

    model.eval()
    projected_denoised = denoise(model, normalized)
    projected_denoised = projected_denoised * (max_vals - min_vals) + min_vals
    back_projected = torch.mm(
        basis, projected_denoised.squeeze(0).reshape(config.spectral_dim, -1)
    ).reshape(channels, height, width).unsqueeze(0)

    elapsed = time.perf_counter() - start
    per_band = config.metric_mode == "per_band"
    metrics = {
        "noisy": evaluate_pair(clean_img, noisy_img, per_band),
        "denoised": evaluate_pair(clean_img, back_projected, per_band),
        "runtime_seconds": elapsed,
        "trainable_parameters": trainable,
    }
    save_json(metrics, output / "metrics.json")
    save_json(history, output / "loss_history.json")

    if config.save_checkpoint:
        torch.save({
            "model_state_dict": model.state_dict(),
            "basis": basis.detach().cpu(),
            "projection_min": min_vals.detach().cpu(),
            "projection_max": max_vals.detach().cpu(),
            "config": config.to_dict(),
        }, output / "checkpoint.pth")

    save_reconstruction_figure(clean_img, noisy_img, back_projected, config.rgb_bands,
                               metrics, output / "reconstruction.png", config.show_plots)
    print(f"Runtime: {elapsed:.3f} s")
    for name in ("noisy", "denoised"):
        m = metrics[name]
        print(f"{name:9s}: PSNR={m['psnr']:.4f} dB | SSIM={m['ssim']:.6f} | SAM={m['sam']:.6f}")
    print(f"Results saved to: {output.resolve()}")
    return back_projected, metrics, model
