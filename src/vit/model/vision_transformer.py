import torch
import torch.nn as nn

from vit.model.transformer import Encoder


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


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        image_size,
        patch_size,
        in_channels,
        embed_dim,
        ffn_dim,
        num_heads,
        num_classes,
        dropout,
    ) -> None:
        super().__init__()
        self.patch_embedding_layer = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim, dropout
        )
        self.encoder = Encoder(num_blocks, embed_dim, ffn_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_layer = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor, mask=None):
        """
        Args:
            images: [batch_size, in_channels, H, W]
        """
        x = self.patch_embedding_layer(
            images
        )  # [batch_size, N + 1, embed_dim], N = num_patches ** 2
        x = self.encoder(x, mask=mask)  # [batch_size, N + 1, embed_dim]
        x = self.norm(x)
        cls_token = x[:, 0]  # [batch_size, embed_dim]
        logits = self.cls_layer(cls_token)  # [batch_size, num_classes]
        return logits
