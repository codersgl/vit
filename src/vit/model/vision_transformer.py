import torch
import torch.nn as nn

from vit.data.dataset import PatchEmbedding
from vit.model.transformer import Encoder


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
