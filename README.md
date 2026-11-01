# Hyper-ZSD

**Hyper-ZSD** is a Zero-shot Noise2Noise framework for hyperspectral image denoising. This repository provides a modular implementation of the method, including spectral subspace projection, stripe neighborhood completion, mask-guided residual learning, and consistency-based self-supervision.

> **Reproducibility note:** This repository is a research-code refactoring of the supplied implementation. Before publication, verify all implementation details against the final manuscript and benchmark protocol. In particular, report the exact data preprocessing, noise parameters, random seeds, hardware, software versions, and evaluation conventions used for every table.

## 1. Repository structure

```text
HyperZSD/
├── hyperzsd/
│   ├── config.py          # Experiment configuration
│   ├── data.py            # MATLAB HSI loading and normalization
│   ├── noise.py           # Six synthetic noise protocols
│   ├── operators.py       # SVD, neighborhood completion, pair downsampling
│   ├── models.py          # Hyper-ZSD network
│   ├── losses.py          # Mask-guided and consistency losses
│   ├── metrics.py         # PSNR, SSIM, SAM
│   ├── visualization.py   # Loss curves and RGB composites
│   ├── engine.py          # Training and reconstruction pipeline
│   └── utils.py           # Seeds, device selection, serialization
├── tests/
│   └── test_core.py
├── data/                  # Put datasets here (not tracked by Git)
├── outputs/               # Experiment outputs (not tracked by Git)
├── main.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 2. Installation

We recommend creating an isolated Python environment. Install a PyTorch build compatible with your CUDA version from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
git clone <YOUR-REPOSITORY-URL>
cd HyperZSD

conda create -n hyperzsd python=3.10 -y
conda activate hyperzsd

# Install the appropriate PyTorch build for your machine first.
pip install -r requirements.txt
pip install -e .
```

## 3. Dataset preparation

Place the MATLAB dataset in `data/`. By default, the loader expects:

- File: `data/Salinas_scene.mat`
- MATLAB variable: `O_Img`
- Cube layout: height × width × spectral bands

You can specify a different path and MATLAB variable using command-line arguments. The loader independently min-max normalizes each spectral band, matching the supplied implementation.

## 4. Run Hyper-ZSD

Default experiment:

```bash
python main.py
```

```bash
python main.py --noise Case5 --spectral-dim 6 \
               --tau 0.05 --threshold-rho 0.05 --max-search-distance 3 \
               --output outputs/Salinas
```

Specify the dataset, noise protocol, training epochs, and output directory:

```bash
python main.py \
  --data data/Salinas_scene.mat \
  --data-key O_Img \
  --noise Case5 \
  --epochs 2000 \
  --spectral-dim 6 \
  --seed 2026 \
  --device auto \
  --output outputs/salinas_case5
```

Available noise protocols:

| Protocol | Description |
|---|---|
| Case1 | i.i.d. Gaussian noise |
| Case2 | Non-i.i.d. Gaussian noise |
| Case3 | Non-i.i.d. Gaussian + stripe corruption |
| Case4 | Gaussian + stripe corruption |
| Case5 | Gaussian + salt-and-pepper + stripe corruption |
| Case6 | Mixed non-i.i.d. Gaussian, impulse, varying stripe, and dead-line corruption |

Use `--metric-mode per_band` to additionally report band-wise metric values in `metrics.json`.

## 5. Outputs

Each run writes the following files to its output directory:

| File | Description |
|---|---|
| `config.json` | Complete experiment configuration |
| `metrics.json` | Noisy-input and denoised-output metrics |
| `loss_history.json` | Epoch-wise loss values |
| `loss_curves.png` | Training loss plots |
| `reconstruction.png` | Ground-truth, noisy, and denoised RGB composites |
| `checkpoint.pth` | Model parameters and projection normalization information |

The checkpoint contains the learned network weights, spectral basis, projection normalization statistics, and configuration.

## 6. Tests

Run the basic unit tests from the repository root:

```bash
pytest -q
```

## 7. Method components

The implementation is organized into the following stages:

1. **Spectral subspace projection:** compute the leading left singular vectors of the noisy hyperspectral cube and project the observation into a reduced-dimensional subspace.
2. **Neighborhood completion:** detect low-variation rows and columns and estimate corrupted lines using nearby non-defective lines.
3. **Self-supervised denoising:** train a convolutional residual estimator using paired downsampling, mask-guided residual loss, and consistency loss.
4. **Uncertainty weighting:** combine the two objectives using learnable scalar parameters.
5. **Spectral reconstruction:** invert the projection using the estimated spectral basis and evaluate the reconstructed cube.

## 8. Reproducibility and scientific reporting

For a publication-quality release, we recommend:

- Pinning exact dependency versions and recording the PyTorch/CUDA/GPU environment.
- Providing dataset download instructions and checksums where redistribution is not permitted.
- Publishing the exact configuration files used to generate every main-table and ablation result.
- Reporting mean and standard deviation over multiple independent random seeds when appropriate.
- Keeping training, validation, and test protocols separate and documenting any use of reference images.
- Releasing pretrained checkpoints only when their provenance and training configuration are documented.
- Reporting whether runtime includes data loading, noise generation, preprocessing, training, and reconstruction.
- Avoiding claims of bitwise determinism unless deterministic kernels and all relevant settings are enforced.

## 9. Implementation notes

- The learnable uncertainty parameters are registered in the model constructor so they are included in the optimizer.
- The SVD implementation uses `torch.linalg.svd`.
- The paired downsampler explicitly checks that image height and width are even.
- The code currently follows the supplied single-image training pipeline (batch size 1).
- The default RGB band indices `(70, 20, 10)` assume the original input cube has at least 71 bands. Change them for datasets with fewer bands.

## 10. Citation

If you use this code, please cite the corresponding Hyper-ZSD paper. Replace the placeholder below with the final bibliographic information after acceptance:

```bibtex
@article{hyperzsd,
  title   = {Enabling Efficient Zero-Shot Noise2Noise for Hyperspectral Images},
  author  = {Xie, Xuelin and Lu, Xiliang and Chen, Long},
  journal = {xxx},
  year    = {xxx},
  doi     = {xxx}
}
```

## License

Add the license required by your institution, funder, or publisher before making the repository public. Do not publish third-party datasets or code unless their licenses permit redistribution.
