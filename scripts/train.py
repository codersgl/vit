import argparse
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
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


def main():
    parser = argparse.ArgumentParser(
        description="VIT",
        epilog="train.py --batch_size 8",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=100)

    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--log_dir", type=str, default="runs/transformer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_dir")
    parser.add_argument(
        "--ffn_dim", type=int, default=3072, help="The hidden dims in FeedForwordNet"
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(args)

    train_dataset = CIFAR100ForViT(train=True, transform=get_train_transform())
    valid_dataset = CIFAR100ForViT(train=False, transform=get_val_transform())

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = VisionTransformer(
        num_blocks=args.num_blocks,
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        embed_dim=args.embed_dim,
        ffn_dim=args.ffn_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        dropout=args.dropout,
    )

    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Scheduler with Warmup
    warmup_epochs = 5
    # Linear warmup during warmup_epochs, then CosineAnnealing
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(args.epochs - warmup_epochs)
    )
    # SequentialLR requires pytorch >= 1.10
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stopping = EarlyStopping(
        patience=args.patience, verbose=True, path=f"{args.checkpoint_dir}/best.pth"
    )

    writer = SummaryWriter(log_dir=args.log_dir)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            epoch,
            device,
            max_grad_norm=args.max_grad_norm,
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
