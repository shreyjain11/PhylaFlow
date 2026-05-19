from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.performer_pytorch import FastAttention


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# TokenGT-style parameter initialization
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02 / sqrt(n_layers))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "padding_idx") and module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


# Performer-style multihead attention (FAVOR+)
class MultiheadPerformerAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        performer_nb_features=64,
        performer_generalized_attention=False,
        dropout=0.0,
        bias=True,
        n_layers=12,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.scaling = self.head_dim**-0.5
        self.performer_nb_features = performer_nb_features
        self.performer_generalized_attention = performer_generalized_attention
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.fast_attention = FastAttention(
            self.head_dim,
            nb_features=performer_nb_features,
            generalized_attention=performer_generalized_attention,
            causal=False,
        )
        self.apply(lambda m: init_params(m, n_layers))

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, C]
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # [B, T, C] -> [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) * self.scaling
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        mask_for_fast_attn = None
        if key_padding_mask is not None:

            mask_for_fast_attn = key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # [B, 1, 1, T]

        # FastAttention expects [B, H, T, D]
        out = self.fast_attention(q, k, v, key_padding_mask=mask_for_fast_attn)
        # [B, num_heads, T, head_dim] -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)  # Apply dropout to the output
        return (
            out,
            None,
        )


# Vanilla MHA:
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, n_layers=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.scaling = self.head_dim**-0.5  # Scale factor for Q or for QK^T
        self.dropout = nn.Dropout(dropout)  # This is for attention probabilities
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.apply(lambda m: init_params(m, n_layers))

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, C]
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) * self.scaling
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_weights = attn_weights.masked_fill(mask == True, float("-inf"))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out, attn_probs
