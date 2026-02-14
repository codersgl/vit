from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit.data.dataset import CIFAR100ForViT, get_train_transform, get_val_transform
from vit.model.vision_transformer import VisionTransformer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    epoch: int,
    device: torch.device,
    max_grad_norm: float = 1.0,
):
    model.train()
    total_loss = 0
    count = 0

    progress_bar = tqdm(dataloader, desc=f"Train epoch {epoch}", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        count += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"[Epoch] Total: avg_loss: {total_loss / count:.4f}")

    return total_loss / count


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    epoch: int,
    device: torch.device,
):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0

    progress_bar = tqdm(dataloader, desc=f"Evaluate epoch {epoch}", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        logits: torch.Tensor = model(images)
        loss = loss_fn(logits, labels)
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).sum().item() / pred.size(0)

        total_loss += loss.item()
        total_acc += acc
        count += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

    print(
        f"[Epoch] Total: avg_loss: {total_loss / count:.4f}, avg_acc: {total_acc / count:.4f}"
    )

    return total_loss / count, total_acc / count


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = torch.inf
        self.delta = delta
        self.path = Path(path)
        if not self.path.exists:
            self.path.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


if __name__ == "__main__":
    batch_size = 32
    embed_dim = 768
    patch_size = 4
    image_size = 32
    num_blocks = 12
    ffn_dim = 3072  # 768 * 4
    num_heads = 12
    dropout = 0.1
    lr = 0.0001
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CIFAR100ForViT(train=True, transform=get_train_transform())
    valid_dataset = CIFAR100ForViT(train=False, transform=get_val_transform())
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size, shuffle=False, num_workers=num_workers
    )

    model = VisionTransformer(
        num_blocks=num_blocks,
        image_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        num_classes=100,
        dropout=dropout,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    loss_fn = nn.CrossEntropyLoss()

    # Add gradient clipping to prevent gradient explosion
    max_grad_norm = 1.0

    train_one_epoch(
        model,
        train_dataloader,
        optimizer,
        loss_fn,
        epoch=1,
        device=device,
        max_grad_norm=max_grad_norm,
    )
    evaluate_one_epoch(model, valid_dataloader, loss_fn, epoch=1, device=device)
