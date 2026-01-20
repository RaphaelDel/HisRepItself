from tqdm import tqdm
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils import create_logger, AverageMeter

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def save_checkpoint(model, optimizer, epoch, config, filename, logger):
    logger.info(f'Saving checkpoint to {filename}.')
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(ckpt, os.path.join(config.OUTPUT.ckpt_dir, filename))

def masked_mse(pred, target, mask):
    # mask: same shape as pred/target, True where valid
    diff = (pred - target)**2
    diff = diff[mask]
    return diff.mean()

def prepare_pose_sequence(raw_joints, device):
    pose_3d = raw_joints.to(device)
    B, T, J = pose_3d.shape
    pose_3d = pose_3d.view(B, T, -1, 3)
    return torch.nan_to_num(pose_3d, nan=0.0)

def _encode_pose(pose_encoder, pose_flat):
    if pose_encoder is None:
        return pose_flat
    if any(p.requires_grad for p in pose_encoder.parameters()):
        return pose_encoder(pose_flat)
    with torch.no_grad():
        return pose_encoder(pose_flat)

def _decode_pose(pose_decoder, pose_latent_flat):
    if pose_decoder is None:
        return pose_latent_flat
    if any(p.requires_grad for p in pose_decoder.parameters()):
        return pose_decoder(pose_latent_flat)
    with torch.no_grad():
        return pose_decoder(pose_latent_flat)

def compute_motion_diffusion_loss(
    model,
    pose_encoder,
    poses,          # (B, T, J, 3)
    T_obs,          # int, number of observed frames
    pose_decoder=None,
    local_loss_weight=0.0,
    device=None,
    return_components=False,
):
    """
    Rectified Flow loss on pose-only latent sequence.
    """
    if device is None:
        device = poses.device

    B, T, J, _ = poses.shape
    assert T_obs < T
    T_pred = T - T_obs

    pose_flat = poses.reshape(B * T, -1)  # (B*T, J*3)
    pose_latent = _encode_pose(pose_encoder, pose_flat).reshape(B, T, -1)

    z_past = pose_latent[:, :T_obs, :]   # (B, T_obs, D)
    z_future = pose_latent[:, T_obs:, :] # (B, T_pred, D)

    # ---- Sample prior for future latents ----
    z0_future = torch.randn_like(z_future)  # (B, T_pred, D)
    z1_future = z_future                    # data

    # ---- Sample time t ~ U[0,1] ----
    t = torch.rand(B, device=device)        # (B,)
    t_b = t.view(B, 1, 1)

    # ---- Interpolate along straight path ----
    z_t_future = (1.0 - t_b) * z0_future + t_b * z1_future  # (B, T_pred, D)

    # Build full sequence z_t = [z_past (clean), z_t_future (noisy)]
    z_t = torch.cat([z_past, z_t_future], dim=1)  # (B, T, D)

    # ---- Time mask: 1 for past, 0 for future ----
    time_mask = torch.zeros(B, T, device=device)
    time_mask[:, :T_obs] = 1.0

    # ---- Ground truth velocity for future slots ----
    v_target_future = z1_future - z0_future  # (B, T_pred, D)

    # ---- Model prediction ----
    v_pred = model(z_t, t, time_mask)         # (B, T, D)
    v_pred_future = v_pred[:, T_obs:, :]      # (B, T_pred, D)

    # ---- Loss (MSE on future timesteps only) ----
    global_loss = F.mse_loss(v_pred_future, v_target_future)

    pose_loss = torch.tensor(0.0, device=device)

    if local_loss_weight > 0:
        pred_future_latent = z0_future + v_pred_future              # (B, T_pred, D)
        pose_pred_flat = _decode_pose(
            pose_decoder, pred_future_latent.reshape(B * T_pred, -1)
        )
        pose_gt_flat = poses[:, T_obs:, :, :].reshape(B * T_pred, -1)
        if pose_pred_flat.shape[-1] != pose_gt_flat.shape[-1]:
            raise ValueError("Pose decoder output dim does not match raw pose dimension.")
        pose_pred = pose_pred_flat.view_as(poses[:, T_obs:, :, :])
        pose_gt = poses[:, T_obs:, :, :]

        valid_mask = torch.isfinite(pose_gt).all(dim=-1, keepdim=True)
        pose_diff = (pose_pred - pose_gt) ** 2
        pose_loss = (pose_diff * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    loss = global_loss + local_loss_weight * pose_loss

    if return_components:
        return loss, {
            "global": global_loss.detach(),
            "pose": pose_loss.detach(),
        }

    return loss

@torch.no_grad()
def sample_future_pose(
    model,
    pose_encoder,
    pose_decoder,
    poses_obs,  # (B, T_obs, J, 3)
    T_pred,
    num_steps=20,
):
    """
    Predict future poses with rectified flow.
    Returns:
      pred_pose: (B, T_pred, J, 3)
    """
    device = poses_obs.device
    B, T_obs, J, _ = poses_obs.shape

    # ---- Encode past ----
    pose_lat_past = _encode_pose(pose_encoder, poses_obs.reshape(B * T_obs, -1)).reshape(B, T_obs, -1)
    D = pose_lat_past.size(-1)

    # ---- Initialize future latent from prior ----
    z_future = torch.randn(B, T_pred, D, device=device)
    T_total = T_obs + T_pred

    # ---- Time mask ----
    time_mask = torch.zeros(B, T_total, device=device)
    time_mask[:, :T_obs] = 1.0

    # ---- ODE integration from t=0 to t=1 ----
    dt = 1.0 / num_steps
    t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for k in range(num_steps):
        t = t_vals[k] * torch.ones(B, device=device)  # (B,)
        z_t = torch.cat([pose_lat_past, z_future], dim=1)  # (B, T_total, D)
        v = model(z_t, t, time_mask)                       # (B, T_total, D)
        v_future = v[:, T_obs:, :]                         # (B, T_pred, D)

        z_future = z_future + dt * v_future                # Euler step

    # ---- Decode future poses ----
    pose_pred_flat = _decode_pose(pose_decoder, z_future.reshape(B * T_pred, -1))
    if pose_pred_flat.shape[-1] != poses_obs[:, :1, :, :].reshape(B, -1).shape[-1]:
        raise ValueError("Pose decoder output dim does not match raw pose dimension.")
    pose_pred = pose_pred_flat.reshape(B, T_pred, J, 3)

    return pose_pred

@hydra.main(config_path="config", config_name="configH36m")
def train(cfg: DictConfig):

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    pose_dim = getattr(cfg, "pose_dim", cfg.in_features)
    use_autoencoder = getattr(cfg, "use_pose_autoencoder", True)

    ## Create experiments directory
    os.makedirs(cfg.OUTPUT.ckpt_dir, exist_ok=True)

    logger = create_logger('')
    logger.info("Initializing with config:")
    logger.info(cfg)

    ################################
    # Load data
    ################################

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
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    writer_train = SummaryWriter(f"{cfg.exp_name}_TRAIN")
    writer_valid =  SummaryWriter(f"{cfg.exp_name}_VALID")
    
    ################################
    # Create model, loss, optimizer
    ################################

    rf_model = instantiate(cfg.motionModel).to(cfg.device)
    encoder = decoder = None
    train_autoencoder = False
    if (not use_autoencoder) and hasattr(rf_model, "latent_dim") and rf_model.latent_dim != pose_dim:
        raise ValueError(
            f"MotionDiffusion latent_dim ({rf_model.latent_dim}) must match pose_dim ({pose_dim}) "
            "when running without the pose autoencoder."
        )

    encoder_checkpoint = ""
    if use_autoencoder:
        encoderDecoder = instantiate(cfg.encoderModel).to(cfg.device)
        encoder = encoderDecoder.encoder
        decoder = encoderDecoder.decoder
        encoder_checkpoint = getattr(cfg, "encoderCheckpoint", "")
        if encoder_checkpoint:
            encoderDecoder.load_state_dict(torch.load(encoder_checkpoint)['model'])
            encoderDecoder.requires_grad_(False)
            encoderDecoder.eval()
            logger.info("Pose autoencoder loaded and frozen.")
        else:
            train_autoencoder = True
            logger.info("Pose autoencoder initialized without checkpoint; training jointly.")
    else:
        logger.info(f"Running rectified flow directly on raw poses (dim={pose_dim}, no encoder/decoder).")
    
    optimizer_params = list(rf_model.parameters())
    if train_autoencoder:
        optimizer_params += list(encoderDecoder.parameters())
    optimizer = torch.optim.AdamW(optimizer_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.COSINEANNEALING.T_max, eta_min=cfg.train.COSINEANNEALING.eta_min)

    num_parameters = sum(p.numel() for p in optimizer_params if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")

    obs_len = cfg.data.opt.input_n
    pred_len = cfg.data.opt.output_n
    diffusion_steps = getattr(cfg, "diffusion_steps", 20)
    
    ################################
    # Begin Training 
    ################################
    global_step = 0
    min_val_loss = 1e6
    

    for epoch in range(cfg.train.epochs):
        start_time = time.time()
        dataiter = iter(dataloader_train)

        timer = {"DATA": 0, "FORWARD": 0, "BACKWARD": 0}

        loss_avg = AverageMeter()
        global_loss_avg = AverageMeter()
        pose_loss_avg = AverageMeter()
        val_avg = AverageMeter()

        train_steps =  len(dataloader_train)

        for i in (tqdmbar := tqdm(range(train_steps), total=train_steps)):

            rf_model.train()
            if train_autoencoder:
                encoderDecoder.train()
            optimizer.zero_grad()

            ################################
            # Load a batch of data
            ################################
            start = time.time()

            try:
                raw_joints = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader_train)
                raw_joints = next(dataiter)

            poses = prepare_pose_sequence(raw_joints, cfg.device)
            timer["DATA"] = time.time() - start

            ################################
            # Forward Pass 
            ################################
            start = time.time()
            
            B, T, _, _ = poses.shape
            loss, loss_parts = compute_motion_diffusion_loss(
                rf_model,
                encoder,
                poses,
                T_obs=obs_len,
                pose_decoder=decoder,
                local_loss_weight=getattr(cfg.train, "local_loss_weight", 0.5),
                return_components=True,
            )

            # loss, pred_joints = compute_loss(model, cfg, joints, epoch=epoch)

            timer["FORWARD"] = time.time() - start

            ################################
            # Backward Pass + Optimization
            ################################
            start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step(epoch + i / train_steps)
            current_lr = optimizer.param_groups[0]['lr']
                
            timer["BACKWARD"] = time.time() - start

            ################################
            # Logging 
            ################################

            loss_avg.update(loss.item(), B)
            global_loss_avg.update(loss_parts["global"].item(), B)
            pose_loss_avg.update(loss_parts["pose"].item(), B)
            timer["LOSS"] = loss_avg.avg
            timer["GLOBAL_LOSS"] = global_loss_avg.avg
            timer["POSE_LOSS"] = pose_loss_avg.avg

            #Tqdm bar
            timer_string= " | ".join([f"{key}: {val:.2f}" for key, val in timer.items()])
            tqdmbar.set_description(f"LR : {current_lr} | {timer_string}")
            if cfg.dry_run:
                print("first dry run break")
                break


        ################################
        # Tensorboard logs
        ################################

        global_step += train_steps

        writer_train.add_scalar("train/loss", loss_avg.avg, epoch)
        writer_train.add_scalar("train/loss_global", global_loss_avg.avg, epoch)
        writer_train.add_scalar("train/loss_pose", pose_loss_avg.avg, epoch)
        logger.info(f"Train/loss : {loss_avg.avg}")
        logger.info(f"Train/global_loss : {global_loss_avg.avg}")
        logger.info(f"Train/pose_loss : {pose_loss_avg.avg}")

        #Val steps
        val_steps =  len(dataloader_val)
        for i in (tqdmbar := tqdm(range(val_steps), total=val_steps)):

            rf_model.eval()
            if train_autoencoder:
                encoderDecoder.eval()
            with torch.no_grad():
                ################################
                # Load a batch of data
                ################################
                start = time.time()

                try:
                    raw_joints = next(dataiter)
                except StopIteration:
                    dataiter = iter(dataloader_val)
                    raw_joints = next(dataiter)

                poses = prepare_pose_sequence(raw_joints, cfg.device)
                timer["DATA"] = time.time() - start

                ################################
                # Forward Pass 
                ################################
                start = time.time()
                
                B, T, _, _ = poses.shape
                poses_obs = poses[:, :obs_len, :, :]             # (B, T_obs, J, 3)
                poses_gt_future = poses[:, obs_len:, :, :]       # (B, T_pred, J, 3)

                pred_pose = sample_future_pose(
                    rf_model,
                    encoder,
                    decoder,
                    poses_obs,
                    pred_len,
                    num_steps=diffusion_steps,
                )

                loss = masked_mse(pred_pose, poses_gt_future, mask=~torch.isnan(poses_gt_future))

                val_avg.update(loss.item(), B)
                timer["LOSS"] = val_avg.avg

                timer["FORWARD"] = time.time() - start
                timer_string= " | ".join([f"{key}: {val:.2f}" for key, val in timer.items()])
                tqdmbar.set_description(f"LR : {current_lr} | {timer_string}")
        # val_loss = compute_loss(model, cfg, joints, src_mask, epoch=epoch) evaluate_loss(model, dataloader_val, cfg)
        writer_valid.add_scalar("val/loss", val_avg.avg, epoch)
        logger.info(f"Val/loss : {val_avg.avg}")

        logger.info(f"Epoch : {epoch}, Train Loss : {loss_avg.avg}, Val Loss : {val_avg.avg}")
        
        val_ade = val_avg.avg
        if val_ade < min_val_loss:
            
            min_val_loss = val_ade
            logger.info('------------------------------BEST MODEL UPDATED------------------------------')
            save_checkpoint(rf_model, optimizer, epoch, cfg, 'best_val_checkpoint.pth.tar', logger)
        logger.info('Val ADE: '+ str(val_ade))

        if cfg.dry_run:
            print("second dryrun break")
            break
        print('time for training: ', time.time()-start_time)
        print('epoch ', epoch, ' finished!')
        save_checkpoint(rf_model, optimizer, epoch, cfg, 'last_epoch_checkpoint.pth.tar', logger)

    logger.info("All done.")

if __name__ == "__main__":
    train()
