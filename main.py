import argparse
from pathlib import Path

from hyperzsd.config import Config
from hyperzsd.data import load_mat_hsi
from hyperzsd.noise import add_noise
from hyperzsd.utils import set_seed, get_device
from hyperzsd.engine import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Hyper-ZSD hyperspectral image denoising")
    parser.add_argument("--data", type=str, default=None, help="Path to input .mat file")
    parser.add_argument("--data-key", type=str, default=None, help="MATLAB variable name")
    parser.add_argument("--output", type=str, default=None, help="Experiment output directory")
    parser.add_argument("--noise", type=str, default=None, choices=[f"Case{i}" for i in range(1, 7)])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--spectral-dim", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None,
                        help="Anomaly-mask percentile in (0, 1]. Default: 0.05")
    parser.add_argument("--threshold-rho", type=float, default=None,
                        help="NC low-variance threshold for dead rows/cols. Default: 0.05")
    parser.add_argument("--max-search-distance", type=int, default=None,
                        help="NC neighbor search radius. Default: 3")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--metric-mode", choices=["whole", "per_band"], default=None)
    parser.add_argument("--show-plots", action="store_true",
                        help="Kept for backward compatibility; Agg backend never shows windows.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    for arg_name, cfg_name in [
        ("data", "data_path"), ("data_key", "data_key"), ("output", "output_dir"),
        ("noise", "noise_type"), ("epochs", "max_epoch"), ("spectral_dim", "spectral_dim"),
        ("tau", "tau"), ("threshold_rho", "threshold_rho"),
        ("max_search_distance", "max_search_distance"),
        ("seed", "seed"), ("device", "device"), ("metric_mode", "metric_mode"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)
    # cfg.show_plots 默认就是 True；--show-plots 只是兼容旧命令，不改变行为。
    if args.show_plots:
        cfg.show_plots = True
    cfg.validate()

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Device: {device}")
    clean_img = load_mat_hsi(cfg.data_path, cfg.data_key)
    noisy_img = add_noise(clean_img, cfg.noise_type)
    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(Path(cfg.output_dir) / "config.json")
    run_experiment(clean_img, noisy_img, cfg, device)


if __name__ == "__main__":
    main()