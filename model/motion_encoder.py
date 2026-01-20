import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.half_dim = dim // 2

    def forward(self, t):
        # t: (B,) scalar in [0,1]
        # sinusoidal or simple MLP; here we use sinusoidal+MLP
        device = t.device
        freqs = torch.arange(self.half_dim, device=device).float()
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * freqs / self.half_dim)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half_dim)
        emb = self.lin1(emb)
        emb = F.silu(emb)
        emb = self.lin2(emb)
        return emb  # (B, dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class RectifiedFlowModel(nn.Module):
    """
    Predicts velocity field for future latents on a sequence that interleaves
    coordinate and pose tokens.

    Inputs:
      - z_t: (B, S, D) interleaved coord/pose latents with noisy future, clean past
      - t: (B,) diffusion time in [0,1]
      - time_mask: (B, S) 1 for past (observed), 0 for future (to be predicted)
      - modality_ids: (B, S) 0 for coordinates, 1 for pose
    """
    def __init__(
        self,
        latent_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        coord_in_dim: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.time_embed = TimeEmbedding(self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, max_len=max_seq_len)
        self.modality_embed = nn.Embedding(2, self.latent_dim)  # 0: coord, 1: pose

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=n_heads,
            dim_feedforward=self.latent_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Binary mask embedding (past vs future)
        self.mask_embed = nn.Embedding(2, self.latent_dim)  # 0: future, 1: past

        self.out_proj = nn.Linear(self.latent_dim, self.latent_dim)

        # Coordinate encoder/decoder to align coord tokens with pose latent dimension
        self.coord_encoder = make_mlp(coord_in_dim, self.latent_dim, self.latent_dim, dropout)
        self.coord_decoder = make_mlp(self.latent_dim, coord_in_dim, self.latent_dim, dropout)

    def forward(self, z_t, t, time_mask, modality_ids):
        """
        z_t:      (B, S, D)  latent sequence with noisy future + clean past
        t:        (B,)       time in [0,1]
        time_mask:(B, S)     1 for past (observed), 0 for future (to be predicted)
        modality_ids: (B, S) 0 for coordinates, 1 for pose
        """
        B, S, D = z_t.shape

        # time embedding
        t_embed = self.time_embed(t).unsqueeze(1)  # (B, 1, D)
        z = z_t + t_embed  # broadcast along sequence

        # mask embedding (past/future)
        mask_emb = self.mask_embed(time_mask.long())  # (B, S, D)
        z = z + mask_emb

        # modality embedding (coord vs pose)
        modality_emb = self.modality_embed(modality_ids.long())  # (B, S, D)
        z = z + modality_emb

        # positional encoding
        z = self.pos_enc(z)

        # transformer encoder
        h = self.transformer(z)  # (B, S, D)

        # project to velocity
        v = self.out_proj(h)  # (B, S, D)

        return v


class MotionDiffusion(nn.Module):
    """
    Predicts velocity field for future pose latents only.

    Inputs:
      - z_t: (B, T, D) pose latents with noisy future, clean past
      - t: (B,) diffusion time in [0,1]
      - time_mask: (B, T) 1 for past (observed), 0 for future (to be predicted)
    """
    def __init__(
        self,
        latent_dim: int = 128,
        n_layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.time_embed = TimeEmbedding(self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, max_len=max_seq_len)
        self.mask_embed = nn.Embedding(2, self.latent_dim)  # 0: future, 1: past

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=n_heads,
            dim_feedforward=self.latent_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, z_t, t, time_mask):
        """
        z_t:       (B, T, D) latent sequence with noisy future + clean past
        t:         (B,)      time in [0,1]
        time_mask: (B, T)    1 for past (observed), 0 for future (to be predicted)
        """
        # time embedding
        t_embed = self.time_embed(t).unsqueeze(1)  # (B, 1, D)
        z = z_t + t_embed  # broadcast along sequence

        # mask embedding (past/future)
        mask_emb = self.mask_embed(time_mask.long())  # (B, T, D)
        z = z + mask_emb

        # positional encoding
        z = self.pos_enc(z)

        # transformer encoder
        h = self.transformer(z)  # (B, T, D)

        # project to velocity
        v = self.out_proj(h)  # (B, T, D)

        return v


# Alias for Hydra configs expecting a lowercase target name.
motion_diffusion = MotionDiffusion
