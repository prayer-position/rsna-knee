"""2.5D multi-view attention-MIL model.

slice encoder (shared 2D CNN) -> gated attention over slices per series
-> + slot embedding -> per-target attention over series -> 12 logits.
"""
import timm
import torch
import torch.nn as nn

from common import SLOTS, TARGETS


class GatedAttention(nn.Module):
    """Ilse et al. 2018 gated attention pooling over the second-to-last axis."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h):  # (..., N, C)
        a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h)))  # (..., N, 1)
        return (a.softmax(dim=-2) * h).sum(dim=-2)


class KneeNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, n_slices=16, drop=0.2, grad_ckpt=True):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        if grad_ckpt:
            self.encoder.set_grad_checkpointing(True)
        c = self.encoder.num_features
        self.slice_pos = nn.Parameter(torch.zeros(n_slices, c))
        self.slice_pool = GatedAttention(c)
        self.slot_emb = nn.Embedding(len(SLOTS), c)
        self.norm = nn.LayerNorm(c)
        self.drop = nn.Dropout(drop)
        n_t = len(TARGETS)
        self.target_attn = nn.Linear(c, n_t)  # one attention query per target over the series
        self.head_w = nn.Parameter(torch.randn(n_t, c) * 0.01)
        self.head_b = nn.Parameter(torch.zeros(n_t))

    def forward(self, x, mask):
        # x: (B, V, S, 3, H, W), mask: (B, V) bool
        b, v, s = x.shape[:3]
        feats = torch.zeros(b, v, s, self.encoder.num_features, device=x.device, dtype=x.dtype)
        present = mask.nonzero(as_tuple=True)
        f = self.encoder(x[present].flatten(0, 1))  # only encode the series that exist
        feats = feats.to(f.dtype)
        feats[present] = f.view(-1, s, f.shape[-1])
        series = self.slice_pool(feats + self.slice_pos)  # (B, V, C)
        series = self.norm(series + self.slot_emb.weight)
        scores = self.target_attn(series).float()  # (B, V, T)
        scores = scores.masked_fill(~mask[..., None], -1e4).softmax(dim=1)
        pooled = torch.einsum("bvt,bvc->btc", scores, series.float())  # (B, T, C)
        return (self.drop(pooled) * self.head_w).sum(-1) + self.head_b  # (B, T)
