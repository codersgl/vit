# Vision Transformer (ViT) on CIFAR-100

This repository contains a PyTorch implementation of the Vision Transformer (ViT) model, trained from scratch on the CIFAR-100 dataset. It serves as an algorithm reproduction and study project.

## Features

- **Pure PyTorch Implementation**: Built from scratch without relying on `timm` or other high-level libraries for the model architecture.
- **CIFAR-100 Adaptation**: Handles 32x32 images with smaller patch sizes (4x4) compared to the original ViT (16x16).
- **Training Stability Improvements**:
  - **Proper Weight Initialization**: Uses Truncated Normal initialization for Linear/Conv layers and constant initialization for LayerNorm/Bias, critical for ViT convergence.
  - **Warmup + Cosine Annealing**: Implements Linear Warmup (5 epochs) followed by Cosine Annealing learning rate schedule.
  - **Gradient Clipping**: Prevents exploding gradients during the initial phase of training.
  - **Dropout Regularization**: Applied to attention weights, FFN hidden layers, and residual connections.
- **Modern Tooling**: Uses `uv` for fast Python package management.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (An extremely fast Python package installer and resolver)

## Installation

This project uses `uv` for dependency management.

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

# Base (768 dim, 12 heads) - Default
uv run python scripts/train.py model=vit_base

# Large (1024 dim, 16 heads)
uv run python scripts/train.py model=vit_large
```

### Custom Training Arguments

You can override any configuration parameter using Hydra syntax:

```bash
# Train with custom epochs and batch size
uv run python scripts/train.py \
    training.epochs=100 \
    data.batch_size=128 \
    optimizer.lr=1e-3
```

Common overrides:
- `training.epochs`: Number of training epochs
- `data.batch_size`: Batch size
- `optimizer.lr`: Learning rate
- `model.dropout`: Dropout rate
- `training.patience`: Early stopping patience

### Monitoring

Training logs (loss, accuracy, learning rate) are saved to the `runs/` directory and can be visualized with TensorBoard.

```bash
tensorboard --logdir runs/
```

## Project Structure

```
.
├── checkpoint_dir/     # Saved model checkpoints
├── config/            # Hydra configuration files
│   ├── config.yaml    # Main configuration
│   ├── data/
│   ├── model/         # Model presets (tiny, small, base, large)
│   └── optimizer/
├── data/              # CIFAR-100 dataset (downloaded automatically)
├── runs/              # TensorBoard logs
├── scripts/
│   └── train.py       # Main training script
├── src/
│   └── vit/
│       ├── data/      # Dataset and Patch Embeddings
│       ├── model/     # Transformer & ViT Architecture
│       └── engine.py  # Training & Evaluation loops
├── pyproject.toml     # Project dependencies
└── README.md
```

## Implementation Details

### Initialization
ViT is known to be sensitive to initialization. This implementation uses:
- `nn.init.trunc_normal_` (std=0.02) for Linear and Conv2d weights.
- `nn.init.constant_` (0) for biases.
- `nn.init.constant_` (1.0) for LayerNorm weights.

### Learning Rate Schedule
To ensure stable convergence from scratch:
1. **Warmup**: Linearly increases LR from `1e-5` to `target_lr` over the first 5 epochs.
2. **Cosine Decay**: Decays LR following a cosine curve for the remaining epochs.
