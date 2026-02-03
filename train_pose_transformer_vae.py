from tqdm import tqdm
import numpy as np
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils import create_logger, AverageMeter
from utils import data_utils
from model.pose_transformer_vae import kl_divergence

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def save_checkpoint(model, optimizer, epoch, config, filename, logger):
    logger.info(f"Saving checkpoint to {filename}.")
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, os.path.join(config.OUTPUT.ckpt_dir, filename))


def parse_batch(batch, cfg, num_joints):
    rot = batch[0] if isinstance(batch, (list, tuple)) else batch

    if rot.dim() == 3:
        B, T, D = rot.shape
        if D == num_joints * 3 + 3:
            rot = rot[..., 3:]
        rot = rot.view(B, T, -1, 3)
    elif rot.dim() == 4:
        pass
    else:
        raise ValueError(f"Unexpected rotation tensor shape: {rot.shape}")

    return rot


def expmap_to_xyz(expmap: torch.Tensor) -> torch.Tensor:
    B, J, _ = expmap.shape
    exp = torch.zeros(B, 3 + J * 3, device=expmap.device)
    exp[:, 3:] = expmap.reshape(B, J * 3)
    xyz = data_utils.expmap2xyz_torch(exp)
    return xyz


def geodesic_angle(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean geodesic distance between rotation matrices derived from axis-angle.
    pred/target: (B, J, 3)
    """
    B, J, _ = pred.shape
    pred_flat = pred.reshape(B * J, 3)
    target_flat = target.reshape(B * J, 3)

    R_pred = data_utils.expmap2rotmat_torch(pred_flat)
    R_gt = data_utils.expmap2rotmat_torch(target_flat)
    R_rel = torch.matmul(R_pred, R_gt.transpose(1, 2))
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos)
    return angle.mean()


def compute_losses(model, cfg, pose_expmap):
    recon, mu, logvar = model(pose_expmap)

    xyz_pred = expmap_to_xyz(recon)
    xyz_gt = expmap_to_xyz(pose_expmap)
    l_fk = torch.mean((xyz_pred - xyz_gt) ** 2)

    l_ang = geodesic_angle(recon, pose_expmap)
    recon_loss = cfg.loss.fk * l_fk + cfg.loss.angle * l_ang

    kl = kl_divergence(mu, logvar)
    total = recon_loss + cfg.loss.beta * kl
    return total, recon_loss, l_fk, l_ang, kl


@hydra.main(config_path="config", config_name="configPoseTransformerVAE")
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

    writer_train = SummaryWriter(f"{cfg.exp_name}_TRAIN")
    writer_valid = SummaryWriter(f"{cfg.exp_name}_VALID")

    model = instantiate(cfg.encoderModel).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.COSINEANNEALING.T_max, eta_min=cfg.train.COSINEANNEALING.eta_min
    )

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")

    min_val_loss = 1e6

    for epoch in range(cfg.train.epochs):
        start_time = time.time()
        dataiter = iter(dataloader_train)
        timer = {"DATA": 0, "FORWARD": 0, "BACKWARD": 0}

        loss_avg = AverageMeter()
        fk_avg = AverageMeter()
        ang_avg = AverageMeter()
        kl_avg = AverageMeter()
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

            rot = parse_batch(raw_batch, cfg, cfg.num_joints).to(cfg.device)
            timer["DATA"] = time.time() - start

            B, T, J, D = rot.shape
            rot = rot.view(B * T, J, D)

            start = time.time()
            total, recon_loss, l_fk, l_ang, kl = compute_losses(model, cfg, rot)
            timer["FORWARD"] = time.time() - start

            start = time.time()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step(epoch + i / train_steps)
            current_lr = optimizer.param_groups[0]["lr"]
            timer["BACKWARD"] = time.time() - start

            batch_size = rot.size(0)
            loss_avg.update(total.item(), batch_size)
            fk_avg.update(l_fk.item(), batch_size)
            ang_avg.update(l_ang.item(), batch_size)
            kl_avg.update(kl.item(), batch_size)
            timer["LOSS"] = loss_avg.avg
            timer_string = " | ".join([f"{key}: {val:.4f}" for key, val in timer.items()])
            tqdmbar.set_description(f"E {epoch:03d} | {timer_string} | LR {current_lr:.2e}")

        writer_train.add_scalar("loss/total", loss_avg.avg, epoch)
        writer_train.add_scalar("loss/fk", fk_avg.avg, epoch)
        writer_train.add_scalar("loss/angle", ang_avg.avg, epoch)
        writer_train.add_scalar("loss/kl", kl_avg.avg, epoch)
        logger.info(f"Train total loss: {loss_avg.avg:.6f}")

        if epoch % cfg.train.val_frequency == 0 or epoch == cfg.train.epochs - 1:
            model.eval()
            val_avg = AverageMeter()
            fk_val = AverageMeter()
            ang_val = AverageMeter()
            kl_val = AverageMeter()
            val_steps = len(dataloader_val)
            dataiter = iter(dataloader_val)

            with torch.no_grad():
                for _ in tqdm(range(val_steps), total=val_steps):
                    try:
                        raw_batch = next(dataiter)
                    except StopIteration:
                        dataiter = iter(dataloader_val)
                        raw_batch = next(dataiter)

                    rot = parse_batch(raw_batch, cfg, cfg.num_joints).to(cfg.device)
                    B, T, J, D = rot.shape
                    rot = rot.view(B * T, J, D)

                    total, recon_loss, l_fk, l_ang, kl = compute_losses(model, cfg, rot)
                    batch_size = rot.size(0)
                    val_avg.update(total.item(), batch_size)
                    fk_val.update(l_fk.item(), batch_size)
                    ang_val.update(l_ang.item(), batch_size)
                    kl_val.update(kl.item(), batch_size)

            writer_valid.add_scalar("loss/total", val_avg.avg, epoch)
            writer_valid.add_scalar("loss/fk", fk_val.avg, epoch)
            writer_valid.add_scalar("loss/angle", ang_val.avg, epoch)
            writer_valid.add_scalar("loss/kl", kl_val.avg, epoch)
            logger.info(f"Val total loss: {val_avg.avg:.6f}")

            if val_avg.avg < min_val_loss:
                min_val_loss = val_avg.avg
                save_checkpoint(model, optimizer, epoch, cfg, "best_val_checkpoint.pth.tar", logger)

        save_checkpoint(model, optimizer, epoch, cfg, "last_epoch_checkpoint.pth.tar", logger)
        logger.info(f"Epoch {epoch} done in {(time.time() - start_time):.2f}s.")


if __name__ == "__main__":
    train()
