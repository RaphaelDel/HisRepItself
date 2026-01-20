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

def compute_loss(model, cfg, joints, epoch=0):
    """
    Compute masked reconstruction loss for pose autoencoder.

    Args:
        model: a nn.Module with 'encoder' and 'decoder' attributes
        cfg: configuration object (can hold masking params)
        joints: (B, J, 3) tensor of input 3D joints
        src_mask: (B, J) tensor indicating valid joints (1 = valid, 0 = missing)
        epoch: current training epoch

    Returns:
        loss: MSE loss between original and reconstructed joints
    """
    B, J, C = joints.shape  # typically (batch, joints, 3)
    joints = joints.to(cfg.device)
    mask = ~torch.isnan(joints)          # True where valid
    joints = torch.nan_to_num(joints, nan=0.0)

    recon, latent = model(joints, epoch, return_latent=True)

    # Compute loss only on originally valid joints
    loss = masked_mse(recon, joints, mask)

    return loss, recon


@hydra.main(config_path="config", config_name="configH36m")
def train(cfg: DictConfig):

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ## Create experiments directory
    os.makedirs(cfg.OUTPUT.ckpt_dir, exist_ok=True)

    logger = create_logger('')
    logger.info("Initializing with config:")
    logger.info(cfg)

    ################################
    # Load data
    ################################

    dataset_train = instantiate(cfg.data, split=0) # train split
    logger.info(f"Training on a total of {dataset_train.__len__()} annotations.")
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    dataset_val = instantiate(cfg.data, split=1) # val split
    logger.info(f"Validating on a total of {dataset_val.__len__()} annotations.")
    dataloader_val = DataLoader(dataset_val, batch_size=cfg.test_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    writer_train = SummaryWriter(f"{cfg.exp_name}_TRAIN")
    writer_valid =  SummaryWriter(f"{cfg.exp_name}_VALID")
    
    ################################
    # Create model, loss, optimizer
    ################################

    model = instantiate(cfg.encoderModel).to(cfg.device)

    # if config["MODEL"]["checkpoint"] != "":
    #     logger.info(f"Loading checkpoint from {config['MODEL']['checkpoint']}")
    #     checkpoint = torch.load(os.path.join(config['OUTPUT']['ckpt_dir'], config["MODEL"]["checkpoint"]))
    #     model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.COSINEANNEALING.T_max, eta_min=cfg.train.COSINEANNEALING.eta_min)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")
    
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
        val_avg = AverageMeter()

        train_steps =  len(dataloader_train)

        for i in (tqdmbar := tqdm(range(train_steps), total=train_steps)):

            model.train()
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

            # _, pose_3d = preprocess(raw_joints, cfg)  # train encoder on 3D pose only
            pose_3d = raw_joints.to(cfg.device)
            B, T, J = pose_3d.shape
            pose_3d = pose_3d.view(B, T, -1, 3)  # (B, T, J, 3)

            timer["DATA"] = time.time() - start

            ################################
            # Forward Pass 
            ################################
            start = time.time()
            
            B, T, J, D = pose_3d.shape
            pose_3d = pose_3d.view(B*T, J, D)
            loss, pred_joints = compute_loss(model, cfg, pose_3d, epoch=epoch)

            timer["FORWARD"] = time.time() - start

            ################################
            # Backward Pass + Optimization
            ################################
            start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step(epoch + i / train_steps)
            current_lr = optimizer.param_groups[0]['lr']
                
            timer["BACKWARD"] = time.time() - start

            ################################
            # Logging 
            ################################

            loss_avg.update(loss.item(), len(pose_3d))
            timer["LOSS"] = loss_avg.avg

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
        logger.info(f"Train/loss : {loss_avg.avg}")

        #Val steps
        val_steps =  len(dataloader_val)
        for i in (tqdmbar := tqdm(range(val_steps), total=val_steps)):

            model.eval()
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

                # _, pose_3d = preprocess(raw_joints, cfg)  # train encoder on 3D pose only
                pose_3d = raw_joints.to(cfg.device)
                B, T, J = pose_3d.shape
                pose_3d = pose_3d.view(B, T, -1, 3)  # (B, T, J, 3)

                timer["DATA"] = time.time() - start

                ################################
                # Forward Pass 
                ################################
                start = time.time()
                
                B, T, J, D = pose_3d.shape
                pose_3d = pose_3d.view(B*T, J, D)
                loss, pred_joints = compute_loss(model, cfg, pose_3d, epoch=epoch)

                val_avg.update(loss.item(), len(pose_3d))
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
            save_checkpoint(model, optimizer, epoch, cfg, 'best_val_checkpoint.pth.tar', logger)
        logger.info('Val ADE: '+ str(val_ade))

        if cfg.dry_run:
            print("second dryrun break")
            break
        print('time for training: ', time.time()-start_time)
        print('epoch ', epoch, ' finished!')
        save_checkpoint(model, optimizer, epoch, cfg, 'last_epoch_checkpoint.pth.tar', logger)

    logger.info("All done.")

if __name__ == "__main__":
    train()
