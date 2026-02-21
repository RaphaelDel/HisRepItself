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
from utils.utils import AverageMeter, create_logger


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


def prepare_input_rep(
    rot_expmap: torch.Tensor,
    input_representation: str,
    use_clean_fk: bool,
    data_scale: float = 1.0,
) -> torch.Tensor:
    if data_scale <= 0:
        raise ValueError(f"data_scale must be > 0, got {data_scale}")
    if input_representation == "expmap":
        return rot_expmap
    if input_representation == "xyz":
        xyz = expmap_to_xyz(rot_expmap, use_clean_fk=use_clean_fk)
        return xyz / data_scale
    raise ValueError(f"Unsupported input_representation: {input_representation}")


def rep_to_xyz(
    rep: torch.Tensor,
    input_representation: str,
    use_clean_fk: bool,
) -> torch.Tensor:
    if input_representation == "xyz":
        return rep
    if input_representation == "expmap":
        return expmap_to_xyz(rep, use_clean_fk=use_clean_fk)
    raise ValueError(f"Unsupported input_representation: {input_representation}")


def unscale_xyz_for_metric(
    xyz_in_model_scale: torch.Tensor,
    input_representation: str,
    data_scale: float,
) -> torch.Tensor:
    if input_representation == "xyz":
        return xyz_in_model_scale * data_scale
    return xyz_in_model_scale


def mpjpe_loss(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    dist = torch.norm(pred_xyz - gt_xyz, dim=-1)  # (B, T, J)
    if loss_mask is None:
        return dist.mean()
    if loss_mask.sum().item() == 0:
        return pred_xyz.new_zeros(())
    return (dist * loss_mask).sum() / loss_mask.sum()


def compute_auxformer_nepa_joint_losses(
    outputs: dict[str, torch.Tensor],
    input_representation: str,
    use_clean_fk: bool,
    weights,
    enable_mask_loss: bool = True,
    enable_denoise_loss: bool = True,
):
    pred_next_xyz = rep_to_xyz(outputs["pred_next"], input_representation, use_clean_fk)
    gt_next_xyz = rep_to_xyz(outputs["gt_next"], input_representation, use_clean_fk)
    l_pred = mpjpe_loss(pred_next_xyz, gt_next_xyz)

    l_mask = l_pred.new_zeros(())
    if enable_mask_loss:
        mask_pred_xyz = rep_to_xyz(outputs["mask_pred"], input_representation, use_clean_fk)
        mask_gt_xyz = rep_to_xyz(outputs["mask_gt"], input_representation, use_clean_fk)
        l_mask = mpjpe_loss(mask_pred_xyz, mask_gt_xyz, loss_mask=outputs["mask_loss_mask"])

    l_denoise = l_pred.new_zeros(())
    if enable_denoise_loss:
        denoise_pred_xyz = rep_to_xyz(outputs["denoise_pred"], input_representation, use_clean_fk)
        denoise_gt_xyz = rep_to_xyz(outputs["denoise_gt"], input_representation, use_clean_fk)
        l_denoise = mpjpe_loss(
            denoise_pred_xyz,
            denoise_gt_xyz,
            loss_mask=outputs["denoise_loss_mask"],
        )

    w_pred = float(getattr(weights, "pred", 1.0))
    w_mask = float(getattr(weights, "mask", 1.0)) if enable_mask_loss else 0.0
    w_denoise = float(getattr(weights, "denoise", 1.0)) if enable_denoise_loss else 0.0
    total = w_pred * l_pred + w_mask * l_mask + w_denoise * l_denoise
    return total, l_pred, l_mask, l_denoise


def save_checkpoint(model, optimizer, epoch, config, filename, logger):
    logger.info(f"Saving checkpoint to {filename}.")
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, os.path.join(config.OUTPUT.ckpt_dir, filename))


@hydra.main(config_path="config", config_name="configAuxFormer_nepa_joint")
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
    min_val_mpjpe = 1e9
    past_len = int(cfg.data.opt.input_n)
    future_len = int(cfg.data.opt.output_n)
    total_len = past_len + future_len

    input_representation = str(cfg.auxformer_nepa.input_representation).lower()
    data_scale = float(getattr(cfg.auxformer_nepa, "data_scale", 1.0))
    use_clean_fk = cfg.fk.mode == "clean"
    infer_use_denoised_feedback = bool(cfg.auxformer_nepa.infer_use_denoised_feedback)
    enable_mask_loss = bool(getattr(cfg.loss, "enable_mask", True))
    enable_denoise_loss = bool(getattr(cfg.loss, "enable_denoise", True))
    if data_scale <= 0:
        raise ValueError(f"auxformer_nepa.data_scale must be > 0, got {data_scale}")
    if input_representation != "xyz" and data_scale != 1.0:
        logger.info("auxformer_nepa.data_scale is only applied when input_representation=xyz (current: %s)", input_representation)
    logger.info(
        "Loss toggles | mask=%s denoise=%s | data_scale=%s",
        enable_mask_loss,
        enable_denoise_loss,
        data_scale,
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
            pose_in = prepare_input_rep(
                rot_gt,
                input_representation=input_representation,
                use_clean_fk=use_clean_fk,
                data_scale=data_scale,
            )
            timer["DATA"] = time.time() - start

            start = time.time()
            outputs = model(
                pose_in,
                compute_mask_branch=enable_mask_loss,
                compute_denoise_branch=enable_denoise_loss,
            )
            total, l_pred, l_mask, l_denoise = compute_auxformer_nepa_joint_losses(
                outputs,
                input_representation=input_representation,
                use_clean_fk=use_clean_fk,
                weights=cfg.loss,
                enable_mask_loss=enable_mask_loss,
                enable_denoise_loss=enable_denoise_loss,
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
                    pose_in = prepare_input_rep(
                        rot_gt,
                        input_representation=input_representation,
                        use_clean_fk=use_clean_fk,
                        data_scale=data_scale,
                    )

                    outputs = model(
                        pose_in,
                        compute_mask_branch=enable_mask_loss,
                        compute_denoise_branch=enable_denoise_loss,
                    )
                    total, l_pred, l_mask, l_denoise = compute_auxformer_nepa_joint_losses(
                        outputs,
                        input_representation=input_representation,
                        use_clean_fk=use_clean_fk,
                        weights=cfg.loss,
                        enable_mask_loss=enable_mask_loss,
                        enable_denoise_loss=enable_denoise_loss,
                    )

                    pred_future_rep = model.predict_rollout(
                        pose_in[:, :past_len],
                        future_frames=future_len,
                        use_denoised_feedback=infer_use_denoised_feedback,
                    )
                    pred_future_xyz = rep_to_xyz(
                        pred_future_rep,
                        input_representation=input_representation,
                        use_clean_fk=use_clean_fk,
                    )
                    pred_future_xyz = unscale_xyz_for_metric(
                        pred_future_xyz,
                        input_representation=input_representation,
                        data_scale=data_scale,
                    )
                    gt_future_xyz = expmap_to_xyz(
                        rot_gt[:, past_len:],
                        use_clean_fk=use_clean_fk,
                    )

                    pred_future_m = pred_future_xyz * eval_scale
                    gt_future_m = gt_future_xyz * eval_scale
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

            mpjpe_mean = float("inf")
            if n_samples > 0:
                mpjpe_avg = mpjpe_sum / n_samples
                mpjpe_mean = mpjpe_avg.mean().item()
                writer_valid.add_scalar("val/mpjpe_mean", mpjpe_mean, epoch)
                for t in range(future_len):
                    writer_valid.add_scalar(
                        f"val/mpjpe_t{t + 1:02d}",
                        mpjpe_avg[t].item(),
                        epoch,
                    )

            logger.info(
                "Val/loss : %.6f | Val/mpjpe_mean: %s",
                val_avg.avg,
                f"{mpjpe_mean:.6f}" if np.isfinite(mpjpe_mean) else "nan",
            )

            if val_avg.avg < min_val_loss:
                min_val_loss = val_avg.avg
                logger.info("------------------------------BEST VAL LOSS MODEL UPDATED------------------------------")
                save_checkpoint(model, optimizer, epoch, cfg, "best_val_checkpoint.pth.tar", logger)

            if np.isfinite(mpjpe_mean) and mpjpe_mean < min_val_mpjpe:
                min_val_mpjpe = mpjpe_mean
                logger.info("------------------------------BEST VAL MPJPE MODEL UPDATED------------------------------")
                save_checkpoint(model, optimizer, epoch, cfg, "best_mpjpe_checkpoint.pth.tar", logger)

        if cfg.dry_run:
            break

        logger.info(f"time for training: {time.time() - start_time:.2f}s")
        logger.info(f"epoch {epoch} finished!")
        save_checkpoint(model, optimizer, epoch, cfg, "last_epoch_checkpoint.pth.tar", logger)

    logger.info("All done.")


if __name__ == "__main__":
    train()
