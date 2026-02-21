from __future__ import annotations

import torch
from torch import nn

from model.auxformer_xyz import STTrans


class AuxFormerNEPAJoint(nn.Module):
    """
    AuxFormer-style model with joint-wise next-step prediction.

    - Input/Output representation is generic pose space (e.g., xyz or expmap), shape (B, T, J, D_in)
    - Main branch predicts next-step pose representation per joint (autoregressive-ready)
    - Aux branches:
        1) masked reconstruction
        2) denoising reconstruction
    """

    def __init__(
        self,
        num_joints: int = 32,
        input_dim: int = 3,
        h_dim: int = 64,
        num_heads: int = 8,
        encoder_depth: int = 4,
        mlp_dim: int = 128,
        dim_per_head: int = 32,
        dropout: float = 0.0,
        max_timestep: int = 256,
        add_joint_token: bool = True,
        mask_ratio: float = 0.5,
        regular_masking: bool = False,
        denoise: bool = True,
        noise_dev: float = 0.5,
        part_noise: bool = True,
        part_noise_ratio: float = 0.25,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.max_timestep = max_timestep

        self.add_joint_token = add_joint_token

        self.mask_ratio = mask_ratio
        self.regular_masking = regular_masking

        self.denoise = denoise
        self.noise_dev = noise_dev
        self.part_noise = part_noise
        self.part_noise_ratio = part_noise_ratio

        self.patch_embed = nn.Linear(input_dim, h_dim)
        self.mask_embed = nn.Parameter(torch.randn(h_dim))

        self.pos_embed = nn.Parameter(torch.randn(1, 1, max_timestep, h_dim))
        self.agent_embed = nn.Parameter(torch.randn(1, num_joints, 1, h_dim))

        self.decoder_pos_embed = nn.Embedding(max_timestep, h_dim)
        self.decoder_agent_embed = nn.Embedding(num_joints, h_dim)

        self.pred_head = nn.Linear(h_dim, input_dim)
        self.mask_head = nn.Linear(h_dim, input_dim)
        self.denoise_head = nn.Linear(h_dim, input_dim)

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

    def _project_input(self, all_traj: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(all_traj)

    def _build_aux_mask(self, b: int, n: int, t: int, device, dtype) -> torch.Tensor:
        if self.regular_masking:
            mask = torch.ones((b, n, t), device=device, dtype=dtype)
            n_mask = int(t * self.mask_ratio)
            if n_mask > 0:
                shuffle = torch.rand((b, n, t), device=device).argsort(dim=-1)
                mask_indices = shuffle[:, :, :n_mask]
                mask.scatter_(2, mask_indices, 0.0)
        else:
            mask = torch.ones((b, n * t), device=device, dtype=dtype)
            n_mask = int(n * t * self.mask_ratio)
            if n_mask > 0:
                shuffle = torch.rand((b, n * t), device=device).argsort(dim=-1)
                mask_indices = shuffle[:, :n_mask]
                mask.scatter_(1, mask_indices, 0.0)
            mask = mask.view(b, n, t)
        return mask

    @staticmethod
    def _build_attention_masks(mask: torch.Tensor, causal_temporal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, t = mask.shape
        mask_s = mask.permute(0, 2, 1).contiguous().view(b * t, n)
        mask_s = mask_s[:, :, None] * mask_s[:, None, :]

        mask_t = mask.contiguous().view(b * n, t)
        mask_t = mask_t[:, :, None] * mask_t[:, None, :]

        if causal_temporal:
            causal = torch.tril(torch.ones((t, t), device=mask.device, dtype=mask.dtype))
            mask_t = mask_t * causal[None, :, :]
        return mask_s, mask_t

    def mask_forward(
        self,
        all_trajs: torch.Tensor,
        mask: torch.Tensor,
        causal_temporal: bool = False,
        decoder_use_causal: bool = False,
    ) -> torch.Tensor:
        b, n, t, _ = all_trajs.shape
        device = all_trajs.device
        if t > self.max_timestep:
            raise ValueError(f"Sequence length {t} exceeds max_timestep={self.max_timestep}")

        inverse_mask = 1.0 - mask

        unmask_tokens = self._project_input(all_trajs) * mask[:, :, :, None]
        unmask_tokens = unmask_tokens + self.pos_embed[:, :, :t, :]
        if self.add_joint_token:
            unmask_tokens = unmask_tokens + self.agent_embed[:, :n, :, :]
        unmask_tokens = unmask_tokens * mask[:, :, :, None]

        mask_s, mask_t = self._build_attention_masks(mask, causal_temporal=causal_temporal)

        decoder_mask_s = None
        decoder_mask_t = None
        if decoder_use_causal:
            full_mask = torch.ones_like(mask)
            decoder_mask_s, decoder_mask_t = self._build_attention_masks(
                full_mask,
                causal_temporal=True,
            )

        time_idx = torch.arange(t, device=device).view(1, 1, t).expand(b, n, t)
        joint_idx = torch.arange(n, device=device).view(1, n, 1).expand(b, n, t)

        decoded_tokens = None
        for l in range(len(self.encoder)):
            if l == 0:
                encoded_tokens = self.encoder[l](unmask_tokens, mask_s, mask_t)
                enc_to_dec_tokens = encoded_tokens * mask[:, :, :, None]

                mask_tokens = self.mask_embed.view(1, 1, 1, -1).expand(b, n, t, -1)
                mask_tokens = mask_tokens + self.decoder_pos_embed(time_idx)
                if self.add_joint_token:
                    mask_tokens = mask_tokens + self.decoder_agent_embed(joint_idx)
                mask_tokens = mask_tokens * inverse_mask[:, :, :, None]

                dec_input_tokens = enc_to_dec_tokens + mask_tokens
                if decoder_use_causal:
                    decoded_tokens = self.decoder[l](
                        dec_input_tokens,
                        decoder_mask_s,
                        decoder_mask_t,
                    )
                else:
                    decoded_tokens = self.decoder[l](dec_input_tokens)
            else:
                encoder_input_pad = decoded_tokens * mask[:, :, :, None]
                encoder_output = self.encoder[l](encoder_input_pad, mask_s, mask_t)
                decoded_tokens = (
                    encoder_output * mask[:, :, :, None]
                    + decoded_tokens * inverse_mask[:, :, :, None]
                )
                if decoder_use_causal:
                    decoded_tokens = self.decoder[l](
                        decoded_tokens,
                        decoder_mask_s,
                        decoder_mask_t,
                    )
                else:
                    decoded_tokens = self.decoder[l](decoded_tokens)
        return decoded_tokens

    def forward(
        self,
        pose_seq: torch.Tensor,
        compute_mask_branch: bool = True,
        compute_denoise_branch: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        pose_seq: (B, T, J, D_in)
        """
        if pose_seq.dim() != 4:
            raise ValueError(f"Expected pose_seq (B, T, J, D), got {pose_seq.shape}")
        if pose_seq.size(2) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {pose_seq.size(2)}")
        if pose_seq.size(3) != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {pose_seq.size(3)}")

        all_traj = pose_seq.permute(0, 2, 1, 3).contiguous()  # (B, J, T, D)
        b, n, t, _ = all_traj.shape
        device = all_traj.device
        dtype = all_traj.dtype

        ordinary_mask = torch.ones((b, n, t), device=device, dtype=dtype)
        aux_mask = None
        if compute_mask_branch:
            aux_mask = self._build_aux_mask(b, n, t, device=device, dtype=dtype)

        noise_mask = None
        all_traj_noise = None
        if compute_denoise_branch:
            noise = torch.randn_like(all_traj) * self.noise_dev
            if self.part_noise:
                noise_mask = (torch.rand((b, n, t), device=device) < self.part_noise_ratio).to(dtype=dtype)
                noise = noise * noise_mask[:, :, :, None]
            else:
                noise_mask = torch.ones((b, n, t), device=device, dtype=dtype)
            all_traj_noise = all_traj + noise

        decoded_main = self.mask_forward(
            all_traj,
            ordinary_mask,
            causal_temporal=True,
            decoder_use_causal=True,
        )
        decoded_aux = None
        if compute_mask_branch:
            decoded_aux = self.mask_forward(
                all_traj,
                aux_mask,
                causal_temporal=False,
                decoder_use_causal=False,
            )
        decoded_noise = None
        if compute_denoise_branch:
            decoded_noise = self.mask_forward(
                all_traj_noise,
                ordinary_mask,
                causal_temporal=True,
                decoder_use_causal=True,
            )

        pred_next = self.pred_head(decoded_main[:, :, :-1, :])  # (B, J, T-1, D)
        gt_next = all_traj[:, :, 1:, :]

        out = {
            "pred_next": pred_next.permute(0, 2, 1, 3).contiguous(),
            "gt_next": gt_next.permute(0, 2, 1, 3).contiguous(),
        }

        if compute_mask_branch:
            pred_mask = self.mask_head(decoded_aux[:, :, :-1, :])  # (B, J, T-1, D)
            mask_gt = all_traj[:, :, 1:, :]
            mask_loss_mask = (1.0 - aux_mask)[:, :, 1:]  # 1 where loss is applied
            out["mask_pred"] = pred_mask.permute(0, 2, 1, 3).contiguous()
            out["mask_gt"] = mask_gt.permute(0, 2, 1, 3).contiguous()
            out["mask_loss_mask"] = mask_loss_mask.permute(0, 2, 1).contiguous()

        if compute_denoise_branch:
            pred_denoise = self.denoise_head(decoded_noise[:, :, :-1, :])  # (B, J, T-1, D)
            if not self.denoise:
                pred_denoise = all_traj[:, :, 1:, :]
            denoise_gt = all_traj[:, :, 1:, :]
            denoise_loss_mask = noise_mask[:, :, 1:]
            out["denoise_pred"] = pred_denoise.permute(0, 2, 1, 3).contiguous()
            out["denoise_gt"] = denoise_gt.permute(0, 2, 1, 3).contiguous()
            out["denoise_loss_mask"] = denoise_loss_mask.permute(0, 2, 1).contiguous()

        return out

    @torch.no_grad()
    def predict_next(
        self,
        pose_context: torch.Tensor,
        use_denoised_feedback: bool = False,
    ) -> torch.Tensor:
        """
        pose_context: (B, T_ctx, J, D_in)
        returns next frame representation: (B, J, D_in)
        """
        if pose_context.dim() != 4:
            raise ValueError(f"Expected pose_context (B, T, J, D), got {pose_context.shape}")
        if pose_context.size(2) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {pose_context.size(2)}")
        if pose_context.size(3) != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {pose_context.size(3)}")

        if pose_context.size(1) > self.max_timestep:
            pose_context = pose_context[:, -self.max_timestep :, :, :]

        all_traj = pose_context.permute(0, 2, 1, 3).contiguous()
        b, n, t, _ = all_traj.shape
        mask = torch.ones((b, n, t), device=all_traj.device, dtype=all_traj.dtype)

        decoded_main = self.mask_forward(
            all_traj,
            mask,
            causal_temporal=True,
            decoder_use_causal=True,
        )
        next_token = decoded_main[:, :, -1:, :]
        next_pred = self.pred_head(next_token).squeeze(2)

        if use_denoised_feedback and self.denoise:
            next_pred = self.denoise_head(next_token).squeeze(2)
        return next_pred

    @torch.no_grad()
    def predict_rollout(
        self,
        pose_past: torch.Tensor,
        future_frames: int,
        use_denoised_feedback: bool = False,
    ) -> torch.Tensor:
        """
        pose_past: (B, T_past, J, D_in)
        returns:   (B, future_frames, J, D_in)
        """
        if future_frames <= 0:
            return pose_past[:, :0]

        seq = pose_past
        future_preds = []
        for _ in range(future_frames):
            next_pred = self.predict_next(
                seq,
                use_denoised_feedback=use_denoised_feedback,
            )  # (B, J, D)
            next_pred = next_pred.unsqueeze(1)  # (B, 1, J, D)
            seq = torch.cat([seq, next_pred], dim=1)
            future_preds.append(next_pred)
        return torch.cat(future_preds, dim=1)

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
        Compatibility wrapper with existing eval scripts style.
        """
        del sample_latent, jepa_pose_past
        past = decoder_pose_past if decoder_pose_past is not None else pose_past
        return self.predict_rollout(
            past,
            future_frames=future_frames,
        )


auxFormerNEPAJoint = AuxFormerNEPAJoint
