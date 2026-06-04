# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mult)
        self.fc2 = nn.Linear(d_model * mult, d_model)
        self.act = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias)
        self.ffn = FeedForward(d_model, drop_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, causal_mask=causal_mask)
        x = self.dropout(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + x

        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            drop_rate=config["drop_rate"],
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(config["emb_dim"], config["n_heads"], config["drop_rate"], config["qkv_bias"])
            for _ in range(config["n_layers"])
        ])
        self.norm = LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            B, T, V = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
            )
            return loss, logits

        return logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
