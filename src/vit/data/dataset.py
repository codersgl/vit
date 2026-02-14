from typing import Tuple

import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset
from vit.model.vision_transformer import PatchEmbedding


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
