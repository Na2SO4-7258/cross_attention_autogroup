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
        attention = logits.softmax(-1).mean(dim=1)
        if values is None:
            context = attention @ reference
        else:
            # Restore the key grid before multiplying the original encoded V.
            # Restore the query grid afterwards: interpolation is linear, so this
            # equals lifting both axes of A before A @ V without allocating N*N.
            if key_size is None or query_size is None:
                raise ValueError("Spatial grid sizes are required for full-resolution values")
            lifted = F.interpolate(attention.reshape(batch, current_tokens, *key_size),
                                   size=values.shape[-2:], mode="bilinear", align_corners=False).flatten(2)
            lifted = lifted / lifted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            context = lifted @ values.flatten(2).transpose(1, 2)
            context = context.transpose(1, 2).reshape(batch, values.shape[1], *query_size)
        result = (context, attention)
        return (*result, scores) if return_scores else result
