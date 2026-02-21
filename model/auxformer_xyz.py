from __future__ import annotations

import os
import random
from typing import Literal

import numpy as np
import torch
from torch import nn

from model.pose_transformer_vae_jepa import PoseTransformerVAEJEPA


class PreNorm(nn.Module):
    def __init__(self, dim: int, net: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dim_per_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, l, _ = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        if mask is not None:
            mask = mask[:, None, :, :].to(dtype=attn.dtype)
            attn = attn * mask
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).reshape(b, l, -1)
        return self.out(z)


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        depth: int = 1,
        num_heads: int = 8,
        dim_per_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(
                                dim,
                                num_heads=num_heads,
                                dim_per_head=dim_per_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x, mask=mask)
            x = x + norm_ffn(x)
        return x


class STTrans(nn.Module):
    """
    Alternating temporal and spatial self-attention on (B, J, T, D) tokens.
    """

    def __init__(
        self,
        h_dim: int,
        depth: int = 1,
        num_heads: int = 8,
        mlp_dim: int = 128,
        dim_per_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        self.transformer_t = nn.ModuleList(
            [
                Transformer(
                    h_dim,
                    mlp_dim,
                    depth=1,
                    num_heads=num_heads,
                    dim_per_head=dim_per_head,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.transformer_s = nn.ModuleList(
            [
                Transformer(
                    h_dim,
                    mlp_dim,
                    depth=1,
                    num_heads=num_heads,
                    dim_per_head=dim_per_head,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask_s: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
        cond_tokens: torch.Tensor | None = None,
        cond_mode: Literal["none", "add", "film"] = "none",
    ) -> torch.Tensor:
        b, n, _, _ = x.shape
        for i in range(self.depth):
            x = x.view(b * n, -1, x.shape[-1])
            x = self.transformer_t[i](x, mask_t)
            x = x.view(b, n, -1, x.shape[-1]).permute(0, 2, 1, 3).contiguous()
            if cond_tokens is not None:
                if cond_mode == "add":
                    x = x + cond_tokens[:, :, None, :]
                elif cond_mode == "film":
                    gamma, beta = cond_tokens.chunk(2, dim=-1)
                    x = x * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]
                elif cond_mode != "none":
                    raise ValueError(f"Unsupported conditioning mode: {cond_mode}")
            x = x.reshape(-1, n, x.shape[-1])
            x = self.transformer_s[i](x, mask_s)
            x = x.view(b, -1, n, x.shape[-1]).permute(0, 2, 1, 3).contiguous()
        return x


class AuxFormerXYZ(nn.Module):
    """
    AuxFormer adapted for pipeline-native XYZ tensors:
    - inputs: (B, T, J, 3)
    - outputs: future prediction + masked reconstruction + denoising branch
    """

    def __init__(
        self,
        num_joints: int = 32,
        past_timestep: int = 50,
        future_timestep: int = 25,
        h_dim: int = 64,
        decoder_dim: int = 64,
        num_heads: int = 8,
        encoder_depth: int = 4,
        mlp_dim: int = 128,
        dim_per_head: int = 32,
        dropout: float = 0.0,
        mask_ratio: float = 0.5,
        range_mask_ratio: bool = False,
        regular_masking: bool = False,
        same_head: bool = False,
        multi_output: bool = False,
        multi_same_head: bool = False,
        decoder_masking: bool = False,
        pred_all: bool = False,
        only_recons_past: bool = False,
        add_joint_token: bool = True,
        add_residual: bool = True,
        concat_vel: bool = False,
        concat_acc: bool = False,
        noise_dev: float = 0.8,
        part_noise: bool = True,
        part_noise_ratio: float = 0.25,
        denoise: bool = True,
        denoise_mode: Literal["all", "past", "future"] = "past",
        range_noise_dev: bool = False,
        use_jepa_conditioning: bool = False,
        conditioning_mode: Literal["none", "add", "film"] = "film",
        conditioning_on_decoder: bool = False,
        conditioning_zero_init: bool = True,
        jepa_ckpt: str | None = None,
        jepa_ckpt_strict: bool = True,
        jepa_freeze: bool = True,
        jepa_sample_latent: bool = False,
        jepa_d_model: int = 256,
        jepa_latent_dim: int = 256,
        jepa_n_heads: int = 8,
        jepa_n_layers: int = 4,
        jepa_dropout: float = 0.1,
        jepa_pred_layers: int = 4,
        jepa_pred_heads: int = 8,
        jepa_pred_dropout: float = 0.1,
        jepa_max_frames: int = 2048,
        jepa_use_vel_ae: bool = False,
        jepa_use_mask_embed: bool = True,
        jepa_use_rope: bool = False,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.past_timestep = past_timestep
        self.future_timestep = future_timestep
        self.all_timesteps = past_timestep + future_timestep

        self.mask_ratio = mask_ratio
        self.range_mask_ratio = range_mask_ratio
        self.regular_masking = regular_masking

        self.same_head = same_head
        self.multi_output = multi_output
        self.multi_same_head = multi_same_head
        self.decoder_masking = decoder_masking

        self.pred_all = pred_all
        self.only_recons_past = only_recons_past
        self.add_joint_token = add_joint_token
        self.add_residual = add_residual

        self.concat_vel = concat_vel
        self.concat_acc = concat_acc

        self.noise_dev = noise_dev
        self.part_noise = part_noise
        self.part_noise_ratio = part_noise_ratio
        self.denoise = denoise
        self.denoise_mode = denoise_mode
        self.range_noise_dev = range_noise_dev
        self.use_jepa_conditioning = use_jepa_conditioning and conditioning_mode != "none"
        self.conditioning_mode = conditioning_mode
        self.conditioning_on_decoder = conditioning_on_decoder
        self.jepa_sample_latent = jepa_sample_latent
        self.jepa_freeze = jepa_freeze

        if decoder_dim != h_dim:
            raise ValueError(f"AuxFormer requires decoder_dim==h_dim, got {decoder_dim} and {h_dim}")

        in_dim = 3
        if self.concat_acc:
            in_dim = 9
        elif self.concat_vel:
            in_dim = 6

        self.patch_embed = nn.Linear(in_dim, h_dim)
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))

        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.all_timesteps, h_dim))
        self.agent_embed = nn.Parameter(torch.randn(1, num_joints, 1, h_dim))

        self.decoder_pos_embed = nn.Embedding(self.all_timesteps, decoder_dim)
        self.decoder_agent_embed = nn.Embedding(num_joints, decoder_dim)

        if self.multi_output:
            self.head = nn.ModuleList([nn.Linear(decoder_dim, 3) for _ in range(encoder_depth)])
        else:
            self.head = nn.Linear(decoder_dim, 3)

        if not self.same_head:
            self.aux_head = nn.Linear(decoder_dim, 3)
            self.aux_head_2 = nn.Linear(decoder_dim, 3)

        self.encoder = nn.ModuleList(
            [
                STTrans(
                    h_dim=h_dim,
                    depth=1,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dim_per_head=dim_per_head,
                    dropout=dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                STTrans(
                    h_dim=h_dim,
                    depth=1,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dim_per_head=dim_per_head,
                    dropout=dropout,
                )
                for _ in range(encoder_depth)
            ]
        )

        self.jepa_encoder: PoseTransformerVAEJEPA | None = None
        self.cond_proj_encoder: nn.ModuleList | None = None
        self.cond_proj_decoder: nn.ModuleList | None = None
        self.jepa_input_dim = 6 if jepa_use_vel_ae else 3
        if self.use_jepa_conditioning:
            self.jepa_encoder = PoseTransformerVAEJEPA(
                num_joints=num_joints,
                d_model=jepa_d_model,
                latent_dim=jepa_latent_dim,
                n_heads=jepa_n_heads,
                n_layers=jepa_n_layers,
                dropout=jepa_dropout,
                pred_layers=jepa_pred_layers,
                pred_heads=jepa_pred_heads,
                pred_dropout=jepa_pred_dropout,
                max_frames=jepa_max_frames,
                use_vel_ae=jepa_use_vel_ae,
                use_mask_embed=jepa_use_mask_embed,
                use_rope=jepa_use_rope,
            )
            if jepa_ckpt:
                self._load_jepa_checkpoint(jepa_ckpt, strict=jepa_ckpt_strict)
            self.jepa_encoder.target_ae.eval()
            if self.jepa_freeze:
                for p in self.jepa_encoder.parameters():
                    p.requires_grad_(False)

            cond_dim = h_dim if self.conditioning_mode == "add" else 2 * h_dim
            self.cond_proj_encoder = nn.ModuleList(
                [nn.Linear(jepa_latent_dim, cond_dim) for _ in range(encoder_depth)]
            )
            if self.conditioning_on_decoder:
                self.cond_proj_decoder = nn.ModuleList(
                    [nn.Linear(jepa_latent_dim, cond_dim) for _ in range(encoder_depth)]
                )
            if conditioning_zero_init:
                for proj in self.cond_proj_encoder:
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                if self.cond_proj_decoder is not None:
                    for proj in self.cond_proj_decoder:
                        nn.init.zeros_(proj.weight)
                        nn.init.zeros_(proj.bias)

    def _load_jepa_checkpoint(self, ckpt_path: str, strict: bool = True) -> None:
        if self.jepa_encoder is None:
            raise RuntimeError("JEPA encoder is not initialized.")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"JEPA checkpoint not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        self.jepa_encoder.load_state_dict(state_dict, strict=strict)

    def _compute_jepa_condition(
        self,
        jepa_pose_past: torch.Tensor,
        future_frames: int,
    ) -> torch.Tensor | None:
        if not self.use_jepa_conditioning:
            return None
        if self.jepa_encoder is None:
            raise RuntimeError("JEPA conditioning is enabled but no JEPA encoder is available.")
        if jepa_pose_past is None:
            raise ValueError("JEPA conditioning is enabled but jepa_pose_past was not provided.")
        if jepa_pose_past.dim() != 4:
            raise ValueError(f"Expected jepa_pose_past (B, T, J, D), got {jepa_pose_past.shape}")
        if jepa_pose_past.size(2) != self.num_joints:
            raise ValueError(
                f"Expected JEPA conditioning with {self.num_joints} joints, got {jepa_pose_past.size(2)}"
            )
        if jepa_pose_past.size(-1) != self.jepa_input_dim:
            raise ValueError(
                f"Expected JEPA conditioning input dim {self.jepa_input_dim}, got {jepa_pose_past.size(-1)}"
            )

        self.jepa_encoder.target_ae.eval()
        if self.jepa_freeze:
            self.jepa_encoder.eval()
            with torch.no_grad():
                z_context = self.jepa_encoder.encode_context(
                    jepa_pose_past, sample_latent=self.jepa_sample_latent
                )
                z_pred, _ = self.jepa_encoder.predict(z_context, future_frames=future_frames)
        else:
            z_context = self.jepa_encoder.encode_context(
                jepa_pose_past, sample_latent=self.jepa_sample_latent
            )
            z_pred, _ = self.jepa_encoder.predict(z_context, future_frames=future_frames)
        return z_pred

    def _sample_mask_ratio(self) -> float:
        if self.range_mask_ratio:
            return random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        return self.mask_ratio

    def _sample_noise_dev(self) -> float:
        if self.range_noise_dev:
            return float(np.random.uniform(low=0.1, high=1.0))
        return self.noise_dev

    def _project_input(self, all_traj: torch.Tensor) -> torch.Tensor:
        if self.concat_acc:
            vel = torch.zeros_like(all_traj)
            vel[:, :, 1:] = all_traj[:, :, 1:] - all_traj[:, :, :-1]
            acc = torch.zeros_like(vel)
            acc[:, :, 1:] = vel[:, :, 1:] - vel[:, :, :-1]
            all_traj_in = torch.cat([all_traj, vel, acc], dim=-1)
        elif self.concat_vel:
            vel = torch.zeros_like(all_traj)
            vel[:, :, 1:] = all_traj[:, :, 1:] - all_traj[:, :, :-1]
            all_traj_in = torch.cat([all_traj, vel], dim=-1)
        else:
            all_traj_in = all_traj
        return self.patch_embed(all_traj_in)

    def _build_aux_mask(self, b: int, n: int, t: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        past_t = min(self.past_timestep, t)
        future_t = max(t - past_t, 0)
        mask_ratio = self._sample_mask_ratio()

        if self.regular_masking:
            mask = torch.ones((b, n, past_t), device=device, dtype=dtype)
            n_mask = int(past_t * mask_ratio)
            if n_mask > 0:
                shuffle = torch.rand((b, n, past_t), device=device).argsort(dim=-1)
                mask_indices = shuffle[:, :, :n_mask]
                mask.scatter_(2, mask_indices, 0.0)
        else:
            mask = torch.ones((b, n * past_t), device=device, dtype=dtype)
            n_mask = int(n * past_t * mask_ratio)
            if n_mask > 0:
                shuffle = torch.rand((b, n * past_t), device=device).argsort(dim=-1)
                mask_indices = shuffle[:, :n_mask]
                mask.scatter_(1, mask_indices, 0.0)
            mask = mask.view(b, n, past_t)

        future_zeros = torch.zeros((b, n, future_t), device=device, dtype=dtype)
        mask_with_no_future = torch.cat([mask, torch.ones_like(future_zeros)], dim=-1)
        mask = torch.cat([mask, future_zeros], dim=-1)
        return mask, mask_with_no_future

    def mask_forward(
        self,
        all_trajs: torch.Tensor,
        mask: torch.Tensor,
        cond_latent: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        b, n, t, _ = all_trajs.shape
        device = all_trajs.device

        inverse_mask = 1.0 - mask

        unmask_tokens = self._project_input(all_trajs) * mask[:, :, :, None]
        unmask_tokens = unmask_tokens + self.pos_embed[:, :, :t, :]
        if self.add_joint_token:
            unmask_tokens = unmask_tokens + self.agent_embed[:, :n, :, :]
        unmask_tokens = unmask_tokens * mask[:, :, :, None]

        mask_s = mask.permute(0, 2, 1).contiguous().view(b * t, n)
        mask_s = mask_s[:, :, None] * mask_s[:, None, :]
        mask_t = mask.contiguous().view(b * n, t)
        mask_t = mask_t[:, :, None] * mask_t[:, None, :]

        time_idx = torch.arange(t, device=device).view(1, 1, t).expand(b, n, t)
        joint_idx = torch.arange(n, device=device).view(1, n, 1).expand(b, n, t)

        out = []
        decoded_tokens = None
        for l in range(len(self.encoder)):
            enc_cond = None
            dec_cond = None
            if cond_latent is not None and self.cond_proj_encoder is not None:
                enc_cond = self.cond_proj_encoder[l](cond_latent)
            if cond_latent is not None and self.cond_proj_decoder is not None:
                dec_cond = self.cond_proj_decoder[l](cond_latent)

            if l == 0:
                encoded_tokens = self.encoder[l](
                    unmask_tokens,
                    mask_s,
                    mask_t,
                    cond_tokens=enc_cond,
                    cond_mode=self.conditioning_mode,
                )
                enc_to_dec_tokens = encoded_tokens * mask[:, :, :, None]

                mask_tokens = self.mask_embed.view(1, 1, 1, -1).expand(b, n, t, -1)
                mask_tokens = mask_tokens + self.decoder_pos_embed(time_idx)
                if self.add_joint_token:
                    mask_tokens = mask_tokens + self.decoder_agent_embed(joint_idx)
                mask_tokens = mask_tokens * inverse_mask[:, :, :, None]

                dec_input_tokens = enc_to_dec_tokens + mask_tokens
                if self.decoder_masking:
                    decoded_tokens = self.decoder[l](
                        dec_input_tokens,
                        1.0 - mask_s,
                        1.0 - mask_t,
                        cond_tokens=dec_cond,
                        cond_mode=self.conditioning_mode,
                    )
                else:
                    decoded_tokens = self.decoder[l](
                        dec_input_tokens,
                        cond_tokens=dec_cond,
                        cond_mode=self.conditioning_mode,
                    )
            else:
                encoder_input_pad = decoded_tokens * mask[:, :, :, None]
                encoder_output = self.encoder[l](
                    encoder_input_pad,
                    mask_s,
                    mask_t,
                    cond_tokens=enc_cond,
                    cond_mode=self.conditioning_mode,
                )
                decoded_tokens = (
                    encoder_output * mask[:, :, :, None]
                    + decoded_tokens * inverse_mask[:, :, :, None]
                )
                if self.decoder_masking:
                    decoded_tokens = self.decoder[l](
                        decoded_tokens,
                        1.0 - mask_s,
                        1.0 - mask_t,
                        cond_tokens=dec_cond,
                        cond_mode=self.conditioning_mode,
                    )
                else:
                    decoded_tokens = self.decoder[l](
                        decoded_tokens,
                        cond_tokens=dec_cond,
                        cond_mode=self.conditioning_mode,
                    )
            out.append(decoded_tokens)

        if self.multi_output and return_all_layers:
            return out
        return decoded_tokens

    def _head_forward(self, tokens: torch.Tensor, head_idx: int | None = None) -> torch.Tensor:
        if self.multi_output:
            if head_idx is None:
                head_idx = -1
            return self.head[head_idx](tokens)
        return self.head(tokens)

    def forward(
        self,
        xyz_seq: torch.Tensor,
        jepa_pose_seq: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        xyz_seq: (B, T, J, 3)
        """
        if xyz_seq.dim() != 4 or xyz_seq.size(-1) != 3:
            raise ValueError(f"Expected xyz_seq (B, T, J, 3), got {xyz_seq.shape}")
        if xyz_seq.size(2) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {xyz_seq.size(2)}")

        all_traj = xyz_seq.permute(0, 2, 1, 3).contiguous()  # (B, J, T, 3)
        b, n, t, _ = all_traj.shape
        device = all_traj.device
        dtype = all_traj.dtype
        past_t = min(self.past_timestep, t)
        future_t = max(t - past_t, 0)

        cond_latent = None
        if self.use_jepa_conditioning:
            cond_latent = self._compute_jepa_condition(
                jepa_pose_past=jepa_pose_seq[:, :past_t] if jepa_pose_seq is not None else None,
                future_frames=future_t,
            )

        start = all_traj[:, :, past_t - 1 : past_t] if past_t > 0 else torch.zeros((b, n, 1, 3), device=device, dtype=dtype)

        ordinary_mask = torch.zeros((b, n, t), device=device, dtype=dtype)
        ordinary_mask[:, :, :past_t] = 1.0

        aux_mask, mask_with_no_future = self._build_aux_mask(b, n, t, device=device, dtype=dtype)

        denoise_mask = torch.zeros((b, n, t), device=device, dtype=dtype)
        denoise_mask[:, :, :past_t] = 1.0

        noise_dev = self._sample_noise_dev()
        noise = torch.randn_like(all_traj) * noise_dev
        if self.part_noise:
            noise_mask = (torch.rand((b, n, t), device=device) < self.part_noise_ratio).to(dtype=dtype)
            noise = noise * noise_mask[:, :, :, None]
        all_traj_noise = all_traj + noise

        if self.multi_output:
            out_layers = self.mask_forward(
                all_traj,
                ordinary_mask,
                cond_latent=cond_latent,
                return_all_layers=True,
            )
            decoded_tokens = out_layers[-1]
        else:
            decoded_tokens = self.mask_forward(all_traj, ordinary_mask, cond_latent=cond_latent)
            out_layers = None
        decoded_tokens_aux = self.mask_forward(all_traj, aux_mask, cond_latent=cond_latent)
        decoded_tokens_noise = self.mask_forward(all_traj_noise, denoise_mask, cond_latent=cond_latent)

        if self.multi_output:
            future_preds = []
            for layer_idx, layer_tokens in enumerate(out_layers):
                head_idx = -1 if self.multi_same_head else layer_idx
                pred_future = self._head_forward(layer_tokens[:, :, past_t:, :], head_idx=head_idx)
                pred_future = pred_future.view(b, n, -1, 3)
                if self.add_residual:
                    pred_future = pred_future + start
                future_preds.append(pred_future.permute(0, 2, 1, 3).contiguous())
            pred_future_out: torch.Tensor | list[torch.Tensor] = future_preds
            pred_future_main = future_preds[-1]
            pred_future_main_bnt = pred_future_main.permute(0, 2, 1, 3).contiguous()
        else:
            pred_future = self._head_forward(decoded_tokens[:, :, past_t:, :])
            pred_future = pred_future.view(b, n, -1, 3)
            if self.add_residual:
                pred_future = pred_future + start
            pred_future_main_bnt = pred_future
            pred_future_out = pred_future.permute(0, 2, 1, 3).contiguous()

        if self.same_head:
            pred_mask = self._head_forward(decoded_tokens_aux, head_idx=-1)
        else:
            pred_mask = self.aux_head(decoded_tokens_aux)
        pred_mask = pred_mask.view(b, n, -1, 3)
        if self.add_residual:
            pred_mask = pred_mask + start

        if self.pred_all:
            loss_mask = torch.ones_like(aux_mask)
        elif self.only_recons_past:
            loss_mask = 1.0 - mask_with_no_future
        else:
            loss_mask = 1.0 - aux_mask
        pred_mask = pred_mask * loss_mask[:, :, :, None]
        masked_gt = all_traj * loss_mask[:, :, :, None]

        if self.denoise_mode == "all":
            denoise_tokens = decoded_tokens_noise
            denoise_target = all_traj
        elif self.denoise_mode == "past":
            denoise_tokens = decoded_tokens_noise[:, :, :past_t, :]
            denoise_target = all_traj[:, :, :past_t, :]
        elif self.denoise_mode == "future":
            denoise_tokens = decoded_tokens_noise[:, :, past_t:, :]
            denoise_target = all_traj[:, :, past_t:, :]
        else:
            raise ValueError(f"Unsupported denoise_mode: {self.denoise_mode}")

        if self.same_head:
            pred_denoise = self._head_forward(denoise_tokens, head_idx=-1)
        else:
            pred_denoise = self.aux_head_2(denoise_tokens)
        pred_denoise = pred_denoise.view(b, n, -1, 3)
        if self.add_residual:
            pred_denoise = pred_denoise + start

        if not self.denoise:
            pred_denoise = denoise_target

        return {
            "future_pred": pred_future_out,
            "future_pred_main": pred_future_main_bnt.permute(0, 2, 1, 3).contiguous(),
            "mask_pred": pred_mask.permute(0, 2, 1, 3).contiguous(),
            "mask_gt": masked_gt.permute(0, 2, 1, 3).contiguous(),
            "mask_loss_mask": loss_mask.permute(0, 2, 1).contiguous(),
            "denoise_pred": pred_denoise.permute(0, 2, 1, 3).contiguous(),
            "denoise_gt": denoise_target.permute(0, 2, 1, 3).contiguous(),
        }

    @torch.no_grad()
    def predict(
        self,
        pose_past: torch.Tensor,
        future_frames: int | None = None,
        jepa_pose_past: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        pose_past: (B, T_past, J, 3) or (B, J, T_past, 3)
        returns future prediction: (B, future_frames, J, 3)
        """
        if pose_past.dim() != 4 or pose_past.size(-1) != 3:
            raise ValueError(f"Expected pose_past rank-4 with last dim 3, got {pose_past.shape}")

        if pose_past.size(2) == self.num_joints:
            pose_past_btjc = pose_past
        elif pose_past.size(1) == self.num_joints:
            pose_past_btjc = pose_past.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Could not infer joint axis for pose_past {pose_past.shape} with num_joints={self.num_joints}"
            )

        b, t_past, j, _ = pose_past_btjc.shape
        if j != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {j}")

        if future_frames is None:
            future_frames = self.future_timestep
        if t_past + future_frames > self.all_timesteps:
            raise ValueError(
                f"Requested sequence length {t_past + future_frames} exceeds model limit {self.all_timesteps}"
            )

        cond_latent = None
        if self.use_jepa_conditioning:
            if jepa_pose_past is None and pose_past_btjc.size(-1) == self.jepa_input_dim:
                jepa_pose_past = pose_past_btjc
            cond_latent = self._compute_jepa_condition(
                jepa_pose_past=jepa_pose_past,
                future_frames=future_frames,
            )

        all_traj = torch.zeros((b, self.num_joints, t_past + future_frames, 3), device=pose_past.device, dtype=pose_past.dtype)
        all_traj[:, :, :t_past, :] = pose_past_btjc.permute(0, 2, 1, 3).contiguous()

        ordinary_mask = torch.zeros((b, self.num_joints, t_past + future_frames), device=pose_past.device, dtype=pose_past.dtype)
        ordinary_mask[:, :, :t_past] = 1.0

        if self.multi_output:
            decoded_tokens = self.mask_forward(
                all_traj,
                ordinary_mask,
                cond_latent=cond_latent,
                return_all_layers=True,
            )[-1]
            pred_future = self._head_forward(decoded_tokens[:, :, t_past:, :], head_idx=-1)
        else:
            decoded_tokens = self.mask_forward(all_traj, ordinary_mask, cond_latent=cond_latent)
            pred_future = self._head_forward(decoded_tokens[:, :, t_past:, :])
        pred_future = pred_future.view(b, self.num_joints, future_frames, 3)

        if self.add_residual and t_past > 0:
            start = all_traj[:, :, t_past - 1 : t_past, :]
            pred_future = pred_future + start
        return pred_future.permute(0, 2, 1, 3).contiguous()

    @torch.no_grad()
    def predict_with_prior(
        self,
        pose_past: torch.Tensor,
        future_frames: int,
        sample_latent: bool = False,
        decoder_pose_past: torch.Tensor | None = None,
        jepa_pose_past: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compatibility wrapper used by existing eval scripts.
        """
        del sample_latent
        past = decoder_pose_past if decoder_pose_past is not None else pose_past
        if past.size(-1) > 3:
            past = past[..., :3]
        return self.predict(
            past,
            future_frames=future_frames,
            jepa_pose_past=jepa_pose_past,
        )


auxFormerXYZ = AuxFormerXYZ
