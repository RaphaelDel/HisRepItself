import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class JointPositionalEmbedding(nn.Module):
    def __init__(self, num_joints: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(num_joints, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos.unsqueeze(0)


def _build_transformer(d_model: int, n_heads: int, n_layers: int, dropout: float) -> nn.Module:
    if n_layers <= 0:
        return nn.Identity()
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class PoseTransformerVAE(nn.Module):
    """
    Transformer VAE for a single pose in expmap (axis-angle) format.

    Input:
      x: (B, J, 3) or (B, J*3)
    Output:
      recon: (B, J, 3)
      mu/logvar: (B, latent_dim)
    """

    def __init__(
        self,
        num_joints: int = 32,
        d_model: int = 256,
        latent_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.joint_embed = nn.Embedding(num_joints, d_model)
        self.in_proj = nn.Linear(3 + d_model, d_model)
        self.enc_pos = JointPositionalEmbedding(num_joints, d_model)
        self.encoder = _build_transformer(d_model, n_heads, n_layers, dropout)
        self.enc_norm = nn.LayerNorm(d_model)

        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        self.z_to_model = nn.Linear(latent_dim, d_model)
        self.dec_pos = JointPositionalEmbedding(num_joints, d_model)
        self.decoder = _build_transformer(d_model, n_heads, n_layers, dropout)
        self.dec_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 3)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.view(x.size(0), self.num_joints, 3)
        B, J, _ = x.shape
        if J != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {J}")

        joint_ids = torch.arange(J, device=x.device)
        joint_emb = self.joint_embed(joint_ids).view(1, J, -1).expand(B, J, -1)
        token = torch.cat([x, joint_emb], dim=-1)

        h = self.in_proj(token)
        h = self.enc_pos(h)
        h = self.encoder(h)
        h = self.enc_norm(h)
        pooled = h.mean(dim=1)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.z_to_model(z).unsqueeze(1).expand(B, self.num_joints, self.d_model)
        h = self.dec_pos(h)
        h = self.decoder(h)
        h = self.dec_norm(h)
        return self.out_proj(h)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        if return_latent:
            return recon, mu, logvar, z
        return recon, mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)


poseTransformerVAE = PoseTransformerVAE
