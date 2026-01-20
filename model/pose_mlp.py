import torch

import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    MLP encoder for poses.

    Architecture:
      - Linear(input_dim -> 256), ReLU
      - Linear(256 -> 256), ReLU
      - Linear(256 -> out_dim)

    Args:
      input_dim: dimensionality of input pose vector (e.g., 66)
      out_dim: dimensionality of final embedding (default 128)
      dropout: dropout probability applied after activations
    """
    def __init__(self, input_dim: int = 66, out_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, 256), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(256, 256), nn.ReLU(inplace=True)])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(256, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim) or any shape where last dimension == input_dim
        returns: (batch, out_dim)
        """
        # flatten all but batch dim
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class MLPDecoder(nn.Module):
    """
    MLP decoder for poses.

    Architecture mirrors MLPEncoder:
      - Linear(input_dim -> 256), ReLU
      - Linear(256 -> 256), ReLU
      - Linear(256 -> out_dim)

    Args:
      input_dim: dimensionality of input embedding (default 128)
      out_dim: dimensionality of reconstructed pose vector (e.g., 66)
      dropout: dropout probability applied after activations
    """
    def __init__(self, input_dim: int = 128, out_dim: int = 66, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, 256), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(256, 256), nn.ReLU(inplace=True)])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(256, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim) or any shape where last dimension == input_dim
        returns: (batch, out_dim)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class PoseMLP(nn.Module):
    """
    Autoencoder-style MLP combining MLPEncoder and MLPDecoder.

    Args:
      pose_dim: dimensionality of the input pose vector
      latent_dim: dimensionality of the latent embedding
      dropout: dropout probability applied inside encoder and decoder
      max_mask_ratio: maximum fraction of joints to mask during training
      device: target device for tensor operations
    """
    def __init__(
        self,
        pose_dim: int = 66,
        latent_dim: int = 128,
        dropout: float = 0.0,
        max_mask_ratio: float = 0.5,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_dim=pose_dim, out_dim=latent_dim, dropout=dropout).to(device)
        self.decoder = MLPDecoder(input_dim=latent_dim, out_dim=pose_dim, dropout=dropout).to(device)
        self.max_mask_ratio = max_mask_ratio
        self.device = device

    def forward(self, x: torch.Tensor, epoch: int, return_latent: bool = False):
        """
        Runs encoder then decoder; optionally returns the latent.

        x: (batch, pose_dim) or any shape where last dimension == pose_dim
        return_latent: when True, also returns the latent embedding
        """
        x = x.to(self.device)
        B, J, C = x.shape  # typically (batch, joints, 3)

        # ===== Apply random masking to joints based on epoch =====
        # Linearly increase masking ratio from 0 → max_mask_ratio over first 100 epochs
        cur_mask_ratio = min(epoch / 100.0, 1.0) * self.max_mask_ratio

        # Random masking: set a fraction of joints to zero
        mask = torch.rand(B, J, device=x.device) < cur_mask_ratio  # True where to mask
        mask = mask.unsqueeze(-1).expand(-1, -1, C)  # (B, J, 3)

        masked_joints = x.clone()
        masked_joints[mask] = 0.0

        # Encode and decode
        # print(" masked_joints shape:", masked_joints.shape)
        masked_flat = masked_joints.view(B, -1)  # (B, J*3)
        # print(" masked_flat shape:", masked_flat.shape)
        latent = self.encoder(masked_flat)      # (B, latent_dim)
        recon_flat = self.decoder(latent)       # (B, J*3)
        out = recon_flat.view(B, J, C)

        if return_latent:
            return out, latent
        return out
