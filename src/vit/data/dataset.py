from typing import Tuple

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset


def get_train_transform(
    mean: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408),
    std: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761),
):
    """Training transform with data augmentation"""
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def get_val_transform(
    mean: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408),
    std: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761),
):
    """Validation transform without data augmentation"""
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return transform


# Backward compatibility
def get_transform(
    mean: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408),
    std: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761),
):
    """Deprecated: Use get_val_transform() instead"""
    return get_val_transform(mean, std)


class CIFAR100ForViT(Dataset):
    def __init__(
        self,
        root="./data",
        train=True,
        transform=None,
    ) -> None:
        super().__init__()
        self.dataset = datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            "Image size must be divisible by patch size."
        )
        self.patch_size = patch_size
        self.image_size = image_size

        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (batch_size, channels, height, width)
        """
        batch_size, _, height, width = images.size()

        assert height == self.image_size and width == self.image_size, (
            f"Input size {height}x{width} doesn't match model ({self.image_size}x{self.image_size})"
        )

        x: torch.Tensor = self.projection(
            images
        )  # [batch_size, embed_dim, self.num_patches, self.num_patches]
        x = x.flatten(
            2
        )  # [batch_size, embed_dim, N], N = self.num_patches * self.num_patches

        x = x.transpose(-1, -2)  # [batch_size, N, embed_dim]

        cls_token = self.cls_token.expand(
            batch_size, -1, -1
        )  # [batch_size, 1, embed_dim]

        x = torch.cat([cls_token, x], dim=1)  # [batch_size, N + 1, embed_dim]

        x = x + self.position_embedding  # [batch_size, N + 1, embed_dim]
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    batch_size = 3
    embed_dim = 768
    patch_size = 4
    image_size = 32

    dataset = CIFAR100ForViT(transform=get_transform())
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    sample = next(iter(dataloader))
    image, label = sample

    print(image.shape)
    print(label.shape)
    patch_embedding = PatchEmbedding(image_size, patch_size, 3, embed_dim)
    x = patch_embedding(image)
    print(x.shape)
