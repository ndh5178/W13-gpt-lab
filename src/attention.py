# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,    #bias는 편향값 현재 기본 code가 False를 주고 있기에 편향값을 사용하지 않는다는 뜻이다.
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        #  qkv projection, output projection, dropout을 정의하세요.
        self.q_proj = nn.Linear(d_model, d_model, bias = qkv_bias)    #nn.Linear 기본형태  nn.Linear(in_features, out_features, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
         multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self. head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self. head_dim).transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1)  # 행렬곱을 맞추기 위한 순서 변경
        attn_scores = attn_scores /math.sqrt(self.head_dim)  # 크기를 줄여서 학습을 안정적으로 sqrt -> 루트

        if causal_mask:
            mask = torch.triu(torch.ones(T, T, device = x.device), diagonal = 1).bool()  #여기서 device = x.device 목적은 하나의 장치에서 통이하기 위함 즉 cpu면 cpu에서 gpu면 gpu에서 장치 통일
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim = -1)  #현재 attention score shape(B, n_heads, T, T)여기서 dim = -1은 마지막 T
        attn_weights = self.dropout(attn_weights) #과적합 방지 랜덤

        context = attn_weights @ v
        context = context.transpose(1, 2)
        context = context.contiguous().view(B, T, C)

        out = self.out_proj(context)

        if return_attention_weights:
            return out, attn_weights
        return out
