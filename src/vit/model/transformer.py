import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, (
            "Embedding dim must be divisible by num_heads"
        )
        self.d_k = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: [batch_size, N, embed_dim]
        """
        batch_size, N, _ = x.size()
        Q: torch.Tensor = self.W_q(x)  # [batch_size, N, embed_dim]
        K: torch.Tensor = self.W_k(x)  # [batch_size, N, embed_dim]
        V: torch.Tensor = self.W_v(x)  # [batch_size, N, embed_dim]

        Q = Q.view(
            batch_size, N, self.num_heads, self.d_k
        )  # [batch_size, N, self.num_heads, self.d_k]
        K = K.view(
            batch_size, N, self.num_heads, self.d_k
        )  # [batch_size, N, self.num_heads, self.d_k]
        V = V.view(
            batch_size, N, self.num_heads, self.d_k
        )  # [batch_size, N, self.num_heads, self.d_k]

        Q = Q.transpose(1, 2)  # [batch_size, self.num_heads, N, d_k]
        K = K.transpose(1, 2)  # [batch_size, self.num_heads, N, d_k]
        V = V.transpose(1, 2)  # [batch_size, self.num_heads, N, d_k]

        multi_attention_score = (
            Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)
        )  # [batch_size, self.num_heads, N, N]

        if mask is not None:
            multi_attention_score = multi_attention_score.masked_fill(
                mask.bool(), -torch.inf
            )

        multi_attention_weight = F.softmax(
            multi_attention_score, dim=-1
        )  # [batch_size, self.num_heads, N, N]

        multi_attention_weight = self.dropout(multi_attention_weight)

        multi_attention = (
            multi_attention_weight @ V
        )  # [batch_size, self.num_heads, N, self.d_k]
        multi_attention = multi_attention.transpose(
            1, 2
        ).contiguous()  # [batch_size, N, self.num_heads, self.d_k]

        attention = multi_attention.view(
            batch_size, N, self.embed_dim
        )  # [batch_size, N, self.embed_dim]
        attention = self.W_o(attention)  # [batch_size, N, self.embed_dim]

        return attention


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, dropout) -> None:
        super().__init__()
        self.multi_attention_layer = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward_layer = FeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        attn_output = self.multi_attention_layer(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        ffn_output = self.feed_forward_layer(self.norm2(x))
        x = x + self.dropout(ffn_output)
        return x


class Encoder(nn.Module):
    def __init__(self, num_blocks, embed_dim, ffn_dim, num_heads, dropout) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            EncoderBlock(embed_dim, ffn_dim, num_heads, dropout)
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
