# Vision Transformer (ViT) on CIFAR-100

This repository contains a PyTorch implementation of the Vision Transformer (ViT) model, trained from scratch on the CIFAR-100 dataset. It serves as an algorithm reproduction and study project using modern Python tooling.

## Features

- **Pure PyTorch Implementation**: Built from scratch without relying on `timm` or other high-level libraries for the model architecture.
- **CIFAR-100 Optimized**: Specifically designed to handle 32x32 images with smaller patch sizes (4x4) compared to the original ViT (16x16).
- **Modern Architecture**:
  - **Pre-LayerNorm Transformer**: Applies LayerNorm *before* attention and FFN blocks for better training stability.
  - **GELU Activation**: Uses Gaussian Error Linear Units instead of ReLU.
  - **Learnable Positional Embeddings**: Adds learnable embeddings to patch tokens.
- **Training Stability Improvements**:
  - **Proper Weight Initialization**: Uses Truncated Normal initialization for Linear/Conv layers and constant initialization for LayerNorm/Bias.
  - **Warmup + Cosine Annealing**: Implements Linear Warmup followed by Cosine Annealing learning rate schedule.
  - **Gradient Clipping**: Prevents exploding gradients (`max_grad_norm=1.0`).
- **Modern Tooling**: Uses `uv` for fast Python package management and `Hydra` for configuration management.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (An extremely fast Python package installer and resolver)
- Python 3.13+ (Managed by `uv`)

## Installation

This project uses `uv` to manage dependencies and virtual environments automatically.

```bash
# Install dependencies
uv sync
```

## Usage

### Training

To start training with the default configuration (ViT-Base):

```bash
uv run python scripts/train.py
```

### Model Selection

We support multiple model sizes via Hydra configuration. You can select a model using the `model` argument:

```bash
# Tiny (192 dim, 3 heads) - Fast training for debugging
uv run python scripts/train.py model=vit_tiny

# Small (384 dim, 6 heads)
uv run python scripts/train.py model=vit_small

# Base (768 dim, 12 heads) - Default configuration
uv run python scripts/train.py model=vit_base

# Large (1024 dim, 16 heads)
uv run python scripts/train.py model=vit_large
```

### Custom Training Arguments

You can override any configuration parameter using Hydra syntax:

```bash
# Train with custom epochs, batch size, and learning rate
uv run python scripts/train.py \
    training.epochs=100 \
    data.batch_size=128 \
    optimizer.lr=1e-3
```

Common overrides:
- `training.epochs`: Number of training epochs
- `data.batch_size`: Batch size
- `optimizer.lr`: Learning rate
- `model.dropout`: Dropout rate (e.g., `model.dropout=0.2`)
- `training.patience`: Early stopping patience (e.g., `training.patience=10`)

### Monitoring

Training logs (loss, accuracy, learning rate) are saved to the `runs/` directory and can be visualized with TensorBoard.

```bash
tensorboard --logdir runs/
```

## Development & Component Testing

This project does not use a formal test runner like `pytest`. Instead, key components have runnable `if __name__ == "__main__":` blocks for quick verification.

```bash
# Test dataset loading and transforms
uv run python src/vit/data/dataset.py

# Test training loop logic (single epoch run)
uv run python src/vit/engine.py
```

## Project Structure

```
.
├── config/                 # Hydra configuration files
│   ├── config.yaml         # Main configuration entry point
│   ├── data/               # Dataset configuration
│   ├── model/              # Model architecture presets (tiny, small, base, large)
│   └── optimizer/          # Optimizer configuration
├── data/                   # CIFAR-100 dataset (downloaded automatically)
├── outputs/                # Hydra output logs (per run)
├── runs/                   # TensorBoard logs
├── scripts/
│   └── train.py            # Main training entry point
├── src/
│   └── vit/
│       ├── data/
│       │   └── dataset.py  # CIFAR-100 dataset wrapper and transforms
│       ├── model/
│       │   ├── transformer.py        # Core transformer blocks (Attention, Encoder)
│       │   └── vision_transformer.py # Main VisionTransformer class and PatchEmbedding
│       └── engine.py       # Training and evaluation loops
├── pyproject.toml          # Project metadata and dependencies
└── README.md
```

## License

This project is open-source and available under the MIT License.
