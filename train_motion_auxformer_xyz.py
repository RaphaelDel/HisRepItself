from __future__ import annotations

import os
import random
import time

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import data_utils
from utils import forward_kinematics as fk_utils
from utils.h36motion3d import x_denorm, x_norm
from utils.utils import AverageMeter, create_logger


def expmap_velocity(expmap_seq: torch.Tensor) -> torch.Tensor:
    """
    Compute expmap velocity from a sequence using relative rotations.
    expmap_seq: (B, T, J, 3)
    returns: (B, T, J, 3) with vel[:, -1] copied from vel[:, -2]
    """
    b, t, j, _ = expmap_seq.shape
    flat = expmap_seq.reshape(b * t * j, 3)
    rot = data_utils.expmap2rotmat_torch(flat).view(b, t, j, 3, 3)

    if t <= 1:
        return torch.zeros_like(expmap_seq)

    rot_prev = rot[:, :-1]
    rot_curr = rot[:, 1:]
    rot_rel = torch.matmul(rot_prev.transpose(-1, -2), rot_curr)

    rel_flat = rot_rel.reshape(-1, 3, 3)
    vel_flat = data_utils.rotmat2expmap_torch(rel_flat)
    vel = torch.zeros_like(expmap_seq)
    vel[:, :-1] = vel_flat.view(b, t - 1, j, 3)
    vel[:, -1] = vel[:, -2]
    return vel


def parse_batch(batch, cfg, num_joints):
    rot = batch[0] if isinstance(batch, (list, tuple)) else batch

    if rot.dim() == 3:
        b, t, d = rot.shape
        if d == num_joints * 3 + 3:
            rot = rot[..., 3:]
        rot = rot.view(b, t, -1, 3)
    elif rot.dim() == 4:
        pass
    else:
        raise ValueError(f"Unexpected rotation tensor shape: {rot.shape}")
    return rot


def clean_fk_from_expmap(expmap):
    parent, offset, _, _ = fk_utils._some_variables()
    n = expmap.shape[0]
    j_n = offset.shape[0]
    device = expmap.device
    p3d_offset = torch.from_numpy(offset).float().to(device).unsqueeze(0).repeat(n, 1, 1)
    angles = expmap[:, 3:].contiguous().view(-1, 3)
    rot = data_utils.expmap2rotmat_torch(angles).view(n, j_n, 3, 3)

    global_rot = []
    global_pos = []
    for i in range(j_n):
        if parent[i] == -1:
            rot_i = rot[:, i]
            pos_i = p3d_offset[:, i]
        else:
            rot_i = torch.matmul(global_rot[parent[i]], rot[:, i])
            pos_i = (
                torch.matmul(global_rot[parent[i]], p3d_offset[:, i].unsqueeze(-1)).squeeze(-1)
                + global_pos[parent[i]]
            )
        global_rot.append(rot_i)
        global_pos.append(pos_i)
    return torch.stack(global_pos, dim=1)


def expmap_to_xyz(rot: torch.Tensor, use_clean_fk: bool = False) -> torch.Tensor:
    b, t, j, _ = rot.shape
    exp = torch.zeros(b * t, 3 + j * 3, device=rot.device, dtype=rot.dtype)
    exp[:, 3:] = rot.reshape(b * t, j * 3)
    xyz = clean_fk_from_expmap(exp) if use_clean_fk else data_utils.expmap2xyz_torch(exp)
    return xyz.view(b, t, j, 3)


def _normalize_pose_cpu(poses: torch.Tensor) -> torch.Tensor:
    if poses.dim() != 4 or poses.size(-1) != 3:
        raise ValueError(f"Expected poses (B, T, J, 3), got {poses.shape}")
    b, t, j, c = poses.shape
    flat = poses.reshape(b, t, j * c).clone().to(torch.float64)
    norm_flat = x_norm(flat)
    return norm_flat.to(poses.dtype).reshape(b, t, j, c)


def _denormalize_pose_cpu(poses: torch.Tensor) -> torch.Tensor:
    if poses.dim() != 4 or poses.size(-1) != 3:
        raise ValueError(f"Expected poses (B, T, J, 3), got {poses.shape}")
    b, t, j, c = poses.shape
    flat = poses.reshape(b, t, j * c).to(torch.float64)
    denorm_flat = x_denorm(flat)
    return denorm_flat.to(poses.dtype).reshape(b, t, j, c)


def infer_eval_scale(cfg: DictConfig, dataset) -> float:
    eval_cfg = getattr(cfg, "eval", None)
    scale = getattr(eval_cfg, "scale", None) if eval_cfg is not None else None
    if scale is not None:
        return float(scale)
    target = str(getattr(cfg.data, "_target_", "")).lower()
    module = dataset.__class__.__module__.lower()
    if "amass" in target or "amass" in module:
        return 1000.0
    return 1.0


def _future_prediction_loss(pred, target):
    """
    pred/target: (B, T_future, J, 3)
    """
    return torch.mean(torch.norm(pred - target, dim=-1))


def _masked_recons_loss(pred, target, mask):
    """
    pred/target: (B, T, J, 3)
    mask: (B, T, J), 1 where loss is applied
    """
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    dist = torch.norm(pred - target, dim=-1)
    return (dist * mask).sum() / mask.sum()


def _denoise_loss(pred, target):
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return torch.mean(torch.norm(pred - target, dim=-1))


def compute_auxformer_losses(
    outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    xyz_gt: torch.Tensor,
    past_len: int,
    weights,
):
    """
    xyz_gt: (B, T, J, 3)
    """
    future_gt = xyz_gt[:, past_len:]
    future_pred = outputs["future_pred"]
    future_main = outputs["future_pred_main"]

    if isinstance(future_pred, list):
        pred_losses = [_future_prediction_loss(p, future_gt) for p in future_pred]
        l_pred = torch.stack(pred_losses).mean()
    else:
        l_pred = _future_prediction_loss(future_pred, future_gt)

    mask_pred = outputs["mask_pred"]
    mask_gt = outputs["mask_gt"]
    mask_loss_mask = outputs["mask_loss_mask"]
    l_mask = _masked_recons_loss(mask_pred, mask_gt, mask_loss_mask)

    denoise_pred = outputs["denoise_pred"]
    denoise_gt = outputs["denoise_gt"]
    l_denoise = _denoise_loss(denoise_pred, denoise_gt)

    w_pred = float(getattr(weights, "pred", 1.0))
    w_mask = float(getattr(weights, "mask", 1.0))
    w_denoise = float(getattr(weights, "denoise", 1.0))
    total = w_pred * l_pred + w_mask * l_mask + w_denoise * l_denoise

    return total, l_pred, l_mask, l_denoise, future_main


def save_checkpoint(model, optimizer, epoch, config, filename, logger):
    logger.info(f"Saving checkpoint to {filename}.")
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, os.path.join(config.OUTPUT.ckpt_dir, filename))


@hydra.main(config_path="config", config_name="configAuxFormer_xyz")
def train(cfg: DictConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.OUTPUT.ckpt_dir, exist_ok=True)

    logger = create_logger("")
    logger.info("Initializing with config:")
    logger.info(cfg)

    dataset_train = instantiate(cfg.data, split=0)
    logger.info(f"Training on a total of {dataset_train.__len__()} annotations.")
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    dataset_val = instantiate(cfg.data, split=1)
    logger.info(f"Validating on a total of {dataset_val.__len__()} annotations.")
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    eval_scale = infer_eval_scale(cfg, dataset_val)

    writer_train = SummaryWriter(f"{cfg.exp_name}_TRAIN")
    writer_valid = SummaryWriter(f"{cfg.exp_name}_VALID")

    model = instantiate(cfg.encoderModel).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.COSINEANNEALING.T_max,
        eta_min=cfg.train.COSINEANNEALING.eta_min,
    )

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {num_parameters}")

    min_val_loss = 1e9
    past_len = int(cfg.data.opt.input_n)
    future_len = int(cfg.data.opt.output_n)
    total_len = past_len + future_len
    use_clean_fk = cfg.fk.mode == "clean"
    use_jepa_conditioning = bool(getattr(getattr(cfg.auxformer, "conditioning", None), "enabled", False))
    jepa_use_vel_ae = bool(
        getattr(getattr(cfg.auxformer, "conditioning", None), "jepa_use_vel_ae", False)
    )

    for epoch in range(cfg.train.epochs):
        start_time = time.time()
        dataiter = iter(dataloader_train)
        timer = {"DATA": 0, "FORWARD": 0, "BACKWARD": 0}

        loss_avg = AverageMeter()
        pred_avg = AverageMeter()
        mask_avg = AverageMeter()
        denoise_avg = AverageMeter()
        train_steps = len(dataloader_train)

        for i in (tqdmbar := tqdm(range(train_steps), total=train_steps)):
            model.train()
            optimizer.zero_grad()

            start = time.time()
            try:
                raw_batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader_train)
                raw_batch = next(dataiter)

            rot_gt = parse_batch(raw_batch, cfg, cfg.num_joints).to(cfg.device)
            rot_gt = rot_gt[:, :total_len]
            xyz_gt_raw = expmap_to_xyz(rot_gt, use_clean_fk=use_clean_fk)
            xyz_gt = _normalize_pose_cpu(xyz_gt_raw.detach().cpu()).to(cfg.device)
            jepa_pose_seq = None
            if use_jepa_conditioning:
                if jepa_use_vel_ae:
                    vel_gt = expmap_velocity(rot_gt)
                    jepa_pose_seq = torch.cat([rot_gt, vel_gt], dim=-1)
                else:
                    jepa_pose_seq = rot_gt
            timer["DATA"] = time.time() - start

            start = time.time()
            outputs = model(xyz_gt, jepa_pose_seq=jepa_pose_seq)
            total, l_pred, l_mask, l_denoise, _ = compute_auxformer_losses(
                outputs,
                xyz_gt,
                past_len=past_len,
                weights=cfg.loss,
            )
            timer["FORWARD"] = time.time() - start

            start = time.time()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step(epoch + i / train_steps)
            current_lr = optimizer.param_groups[0]["lr"]
            timer["BACKWARD"] = time.time() - start

            batch_size = rot_gt.size(0)
            loss_avg.update(total.item(), batch_size)
            pred_avg.update(l_pred.item(), batch_size)
            mask_avg.update(l_mask.item(), batch_size)
            denoise_avg.update(l_denoise.item(), batch_size)
            timer["LOSS"] = loss_avg.avg
            timer_string = " | ".join([f"{k}: {v:.4f}" for k, v in timer.items()])
            tqdmbar.set_description(f"E {epoch:03d} | {timer_string} | LR {current_lr:.2e}")

            if cfg.dry_run:
                break

        writer_train.add_scalar("train/total_loss", loss_avg.avg, epoch)
        writer_train.add_scalar("train/l_pred", pred_avg.avg, epoch)
        writer_train.add_scalar("train/l_mask", mask_avg.avg, epoch)
        writer_train.add_scalar("train/l_denoise", denoise_avg.avg, epoch)
        logger.info(f"Train/loss : {loss_avg.avg}")

        if epoch % cfg.train.val_frequency == 0 or epoch == cfg.train.epochs - 1:
            model.eval()
            val_avg = AverageMeter()
            val_pred_avg = AverageMeter()
            val_mask_avg = AverageMeter()
            val_denoise_avg = AverageMeter()
            mpjpe_sum = torch.zeros(future_len, dtype=torch.float64)
            n_samples = 0

            with torch.no_grad():
                for raw_batch in tqdm(dataloader_val, total=len(dataloader_val)):
                    rot_gt = parse_batch(raw_batch, cfg, cfg.num_joints).to(cfg.device)
                    rot_gt = rot_gt[:, :total_len]
                    xyz_gt_raw = expmap_to_xyz(rot_gt, use_clean_fk=use_clean_fk)
                    xyz_gt = _normalize_pose_cpu(xyz_gt_raw.detach().cpu()).to(cfg.device)
                    jepa_pose_seq = None
                    if use_jepa_conditioning:
                        if jepa_use_vel_ae:
                            vel_gt = expmap_velocity(rot_gt)
                            jepa_pose_seq = torch.cat([rot_gt, vel_gt], dim=-1)
                        else:
                            jepa_pose_seq = rot_gt

                    outputs = model(xyz_gt, jepa_pose_seq=jepa_pose_seq)
                    total, l_pred, l_mask, l_denoise, _ = compute_auxformer_losses(
                        outputs,
                        xyz_gt,
                        past_len=past_len,
                        weights=cfg.loss,
                    )

                    pred_future_norm = model.predict(
                        xyz_gt[:, :past_len],
                        future_frames=future_len,
                        jepa_pose_past=(jepa_pose_seq[:, :past_len] if jepa_pose_seq is not None else None),
                    )
                    pred_future_denorm = _denormalize_pose_cpu(pred_future_norm.detach().cpu()).to(cfg.device)
                    gt_future_denorm = xyz_gt_raw[:, past_len:]

                    pred_future_m = pred_future_denorm * eval_scale
                    gt_future_m = gt_future_denorm * eval_scale
                    mpjpe_t = torch.norm(pred_future_m - gt_future_m, dim=-1).mean(dim=2)

                    batch_size = rot_gt.size(0)
                    val_avg.update(total.item(), batch_size)
                    val_pred_avg.update(l_pred.item(), batch_size)
                    val_mask_avg.update(l_mask.item(), batch_size)
                    val_denoise_avg.update(l_denoise.item(), batch_size)
                    mpjpe_sum += mpjpe_t.detach().cpu().double().sum(dim=0)
                    n_samples += batch_size

                    if cfg.dry_run:
                        break

            writer_valid.add_scalar("val/total_loss", val_avg.avg, epoch)
            writer_valid.add_scalar("val/l_pred", val_pred_avg.avg, epoch)
            writer_valid.add_scalar("val/l_mask", val_mask_avg.avg, epoch)
            writer_valid.add_scalar("val/l_denoise", val_denoise_avg.avg, epoch)

            if n_samples > 0:
                mpjpe_avg = mpjpe_sum / n_samples
                # for t in (2, 4, 8, 10, 25):
                for t in (1, 2, 4, 5, 12):
                    if t <= future_len:
                        writer_valid.add_scalar(f"val/mpjpe_t{t:02d}", mpjpe_avg[t - 1].item(), epoch)
            logger.info(f"Val/loss : {val_avg.avg}")

            if val_avg.avg < min_val_loss:
                min_val_loss = val_avg.avg
                logger.info("------------------------------BEST MODEL UPDATED------------------------------")
                save_checkpoint(model, optimizer, epoch, cfg, "best_val_checkpoint.pth.tar", logger)

        if cfg.dry_run:
            break

        logger.info(f"time for training: {time.time() - start_time:.2f}s")
        logger.info(f"epoch {epoch} finished!")
        save_checkpoint(model, optimizer, epoch, cfg, "last_epoch_checkpoint.pth.tar", logger)

    logger.info("All done.")


if __name__ == "__main__":
    train()
