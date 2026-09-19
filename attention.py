import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim, num_heads=1):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.q = nn.Linear(feature_dim, attention_dim)
        self.k = nn.Linear(feature_dim, attention_dim)
        self.v = nn.Linear(feature_dim, attention_dim)
        self.out = nn.Linear(attention_dim, attention_dim)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, current, reference, return_scores=False, values=None, query_size=None, key_size=None):
        batch, current_tokens, _ = current.shape
        reference_tokens = reference.shape[1]
        q = self.q(current).reshape(batch, current_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(reference).reshape(batch, reference_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2) * self.scale).abs()
        scores = logits.mean(dim=1).amax(dim=-1)
        head_attention = logits.softmax(-1)
        attention = head_attention.mean(dim=1)
        if values is None:
            value_tokens = reference
            transfer_attention = head_attention
        else:
            # Lift each head independently to the original reference grid.
            # Query interpolation commutes with A @ V and the linear projections,
            # so it can run later without allocating a full high-resolution A.
            if key_size is None or query_size is None:
                raise ValueError("Spatial grid sizes are required for full-resolution values")
            value_tokens = values.flatten(2).transpose(1, 2)
            lifted = F.interpolate(head_attention.reshape(batch*self.num_heads, current_tokens, *key_size),
                                   size=values.shape[-2:], mode="bilinear", align_corners=False).flatten(2)
            lifted = lifted / lifted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            transfer_attention = lifted.reshape(batch, self.num_heads, current_tokens, -1)
        v = self.v(value_tokens).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        context = (transfer_attention @ v).transpose(1, 2).reshape(batch, current_tokens, -1)
        context = self.out(context)
        if values is not None:
            context = context.transpose(1, 2).reshape(batch, self.num_heads*self.head_dim, *query_size)
        result = (context, attention)
        return (*result, scores) if return_scores else result
