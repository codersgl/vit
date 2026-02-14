import random
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vit.data.dataset import CIFAR100ForViT, get_train_transform, get_val_transform
from vit.engine import EarlyStopping, evaluate_one_epoch, train_one_epoch
from vit.model.vision_transformer import VisionTransformer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    try:
        if cfg.data.name == "CIFAR100":
            train_dataset = CIFAR100ForViT(
                train=True,
                transform=get_train_transform(
                    mean=cfg.data.normalize_mean, std=cfg.data.normalize_std
                ),
            )
            valid_dataset = CIFAR100ForViT(
                train=False,
                transform=get_val_transform(
                    mean=cfg.data.normalize_mean, std=cfg.data.normalize_std
                ),
            )

        else:
            raise ValueError(f"Unsupported dataset: {cfg.data.name}")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False,
        )

        return train_dataloader, valid_dataloader

    except ImportError as e:
        raise ImportError(f"Required module not found: {e}")
    except AttributeError as e:
        raise AttributeError(f"Missing configuration attribute: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create dataloader for {cfg.data.name}: {e}")


def get_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    try:
        if cfg.optimizer.type == "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.optimizer.lr,
                betas=cfg.optimizer.betas,
                weight_decay=cfg.optimizer.weight_decay,
            )
        elif cfg.optimizer.type == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.optimizer.lr,
                betas=cfg.optimizer.betas,
                weight_decay=cfg.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")
        return optimizer

    except ImportError as e:
        raise ImportError(f"Required module not found: {e}")
    except AttributeError as e:
        raise AttributeError(f"Missing configuration attribute: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create optimizer for {cfg.optimizer.type}: {e}")


def get_model(cfg: DictConfig) -> nn.Module:
    try:
        if cfg.model.name.startswith("vit"):
            return VisionTransformer(
                num_blocks=cfg.model.num_blocks,
                image_size=cfg.model.image_size,
                patch_size=cfg.model.patch_size,
                in_channels=cfg.model.in_channels,
                embed_dim=cfg.model.embed_dim,
                ffn_dim=cfg.model.ffn_dim,
                num_heads=cfg.model.num_heads,
                num_classes=cfg.model.num_classes,
                dropout=cfg.model.dropout,
            )
        else:
            raise ValueError(f"Unsupported model: {cfg.model.name}")

    except ImportError as e:
        raise ImportError(f"Required module not found: {e}")
    except AttributeError as e:
        raise AttributeError(f"Missing configuration attribute: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create model for {cfg.model.name}: {e}")


@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_dataloader, valid_dataloader = get_dataloader(cfg)
    model = get_model(cfg)

    model.to(device)

    optimizer = get_optimizer(model=model, cfg=cfg)

    # Scheduler with Warmup
    warmup_epochs = cfg.training.warmup_epochs

    if warmup_epochs >= cfg.training.epochs:
        # If warmup covers the entire training, only use LinearLR
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=cfg.training.epochs
        )
    else:
        # Linear warmup during warmup_epochs, then CosineAnnealing
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(cfg.training.epochs - warmup_epochs)
        )
        # SequentialLR requires pytorch >= 1.10
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    early_stopping = EarlyStopping(
        patience=cfg.training.patience,
        verbose=True,
        path=f"{cfg.training.checkpoint_dir}/best.pth",
    )

    writer = SummaryWriter(log_dir=cfg.training.log_dir)
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            epoch,
            device,
            max_grad_norm=cfg.model.max_grad_norm,
        )
        valid_loss, valid_acc = evaluate_one_epoch(
            model, valid_dataloader, loss_fn, epoch, device
        )

        scheduler.step()

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print(f"Stopped at epoch {epoch}")
            break

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("valid_loss", valid_loss, epoch)
        writer.add_scalar("valid_acc", valid_acc, epoch)
        writer.add_scalar("learning_rate", current_lr, epoch)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}, lr={current_lr:.6f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
