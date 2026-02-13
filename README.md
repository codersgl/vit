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

To start training with the default configuration (ViT-Base like parameters adapted for CIFAR):

```bash
uv run python scripts/train.py
```

### Custom Training Arguments

You can scale the model size and training parameters using command line arguments:

```bash
# Train a smaller model for testing
uv run python scripts/train.py \
    --epochs 100 \
    --batch_size 128 \
    --embed_dim 192 \
    --num_heads 3 \
    --num_blocks 12 \
    --ffn_dim 768 \
    --learning_rate 1e-3
```

Key arguments:
- `--patch_size`: Size of the patches (default: 4 for 32x32 images)
- `--embed_dim`: Embedding dimension of the transformer
- `--num_heads`: Number of attention heads
- `--num_blocks`: Number of transformer encoder blocks
- `--patience`: Early stopping patience (default: 5)

### Monitoring

Training logs (loss, accuracy, learning rate) are saved to the `runs/` directory and can be visualized with TensorBoard.

```bash
tensorboard --logdir runs/
```

## Project Structure

```
.
├── checkpoint_dir/     # Saved model checkpoints
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
