import csv
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from torch.utils.data import DataLoader

from utils.utils import AverageMeter, create_logger
from train import masked_mse, sample_future_pose

H36M_ACTIONS = [
    "walking",
    "eating",
    "smoking",
    "discussion",
    "directions",
    "greeting",
    "phoning",
    "posing",
    "purchases",
    "sitting",
    "sittingdown",
    "takingphoto",
    "waiting",
    "walkingdog",
    "walkingtogether",
]


def load_encoder_decoder(cfg: DictConfig, device: torch.device, logger):
    if not getattr(cfg, "use_pose_autoencoder", True):
        logger.info("Direct pose diffusion: skipping pose encoder/decoder loading.")
        return None, None

    encoder_decoder = instantiate(cfg.encoderModel).to(device)
    enc_ckpt = torch.load(cfg.encoderCheckpoint, map_location=device)
    encoder_decoder.load_state_dict(enc_ckpt["model"])
    encoder_decoder.eval()
    for p in encoder_decoder.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded pose encoder checkpoint from {cfg.encoderCheckpoint}")
    return encoder_decoder.encoder, encoder_decoder.decoder


def load_motion_model(cfg: DictConfig, device: torch.device, logger):
    rf_model = instantiate(cfg.motionModel).to(device)
    eval_cfg = getattr(cfg, "eval", None)
    ckpt_path = None
    ckpt_path = getattr(eval_cfg, "rf_checkpoint", None)
    if ckpt_path is None:
        ckpt_path = getattr(eval_cfg, "checkpoint", None)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    rf_model.load_state_dict(checkpoint["model"])
    rf_model.eval()
    logger.info(f"Loaded motion predictor checkpoint from {ckpt_path} (epoch={checkpoint.get('epoch', 'n/a')})")
    return rf_model


def prepare_pose_sequence(
    raw_joints: torch.Tensor,
    device: torch.device,
    dims_to_use: Optional[np.ndarray] = None,
) -> torch.Tensor:
    pose_3d = raw_joints.to(device)
    if pose_3d.ndim == 4:
        return torch.nan_to_num(pose_3d, nan=0.0)
    if dims_to_use is not None:
        dims = torch.as_tensor(dims_to_use, device=pose_3d.device)
        pose_3d = torch.index_select(pose_3d, 2, dims)
    B, T, D = pose_3d.shape
    pose_3d = pose_3d.view(B, T, -1, 3)
    return torch.nan_to_num(pose_3d, nan=0.0)


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


def prepare_visual_samples(
    poses_obs: torch.Tensor,
    preds_future: torch.Tensor,
    gt_future: torch.Tensor,
    budget: int,
) -> List[Dict[str, torch.Tensor]]:
    samples: List[Dict[str, torch.Tensor]] = []
    take = min(poses_obs.size(0), budget)
    for idx in range(take):
        obs = poses_obs[idx].detach().cpu()
        pred = preds_future[idx].detach().cpu()
        gt = gt_future[idx].detach().cpu()
        samples.append(
            {
                "pred": torch.nan_to_num(torch.cat([obs, pred], dim=0)),
                "gt": torch.nan_to_num(torch.cat([obs, gt], dim=0)),
            }
        )
    return samples


def save_pose_animation(sequence: torch.Tensor, save_path: str, interval_ms: int, logger):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from utils import viz
    except Exception as exc:
        logger.warning("Visualization skipped (matplotlib not available): %s", exc)
        return

    if sequence.ndim != 3 or sequence.shape[1] != 32:
        logger.warning("Visualization skipped (expected 32 joints, got %s).", tuple(sequence.shape))
        return

    frames = sequence.reshape(sequence.shape[0], -1).numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pose_viz = viz.Ax3DPose(ax)

    def _update(frame_idx):
        pose_viz.update(frames[frame_idx])
        return []

    anim = animation.FuncAnimation(fig, _update, frames=frames.shape[0], interval=interval_ms, blit=False)
    try:
        anim.save(save_path, writer="pillow")
    except Exception as exc:
        logger.warning("Failed to save animation to %s: %s", save_path, exc)
    plt.close(fig)


def save_visualizations(
    samples: List[Dict[str, torch.Tensor]],
    cfg: DictConfig,
    logger,
    scale: float,
    subdir: Optional[str] = None,
):
    if not samples:
        return
    eval_cfg = getattr(cfg, "eval", None)
    save_vis = True
    if eval_cfg is not None:
        if hasattr(eval_cfg, "save_vis"):
            save_vis = bool(eval_cfg.save_vis)
        elif hasattr(eval_cfg, "save_gif"):
            save_vis = bool(eval_cfg.save_gif)
    if not save_vis:
        return
    output_dir = "eval_vis"
    if eval_cfg is not None and hasattr(eval_cfg, "output_dir"):
        output_dir = eval_cfg.output_dir
    interval_ms = 50
    if eval_cfg is not None and hasattr(eval_cfg, "gif_interval"):
        interval_ms = int(eval_cfg.gif_interval)

    save_dir = os.path.join(os.getcwd(), output_dir)
    if subdir:
        save_dir = os.path.join(save_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Saving %d visualization sets to %s", len(samples), save_dir)
    for idx, sample in enumerate(samples):
        stem = f"sample_{idx:02d}"
        gt_motion = sample["gt"] * scale
        pred_motion = sample["pred"] * scale
        save_pose_animation(gt_motion, os.path.join(save_dir, f"{stem}_gt.gif"), interval_ms, logger)
        save_pose_animation(pred_motion, os.path.join(save_dir, f"{stem}_pred.gif"), interval_ms, logger)


def compute_mpjpe_by_timestep(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(gt).all(dim=-1)
    diff = torch.norm(pred - gt, dim=-1)
    diff = diff * valid
    denom = valid.sum(dim=-1).clamp_min(1.0)
    mpjpe = diff.sum(dim=-1) / denom
    return mpjpe.sum(dim=0)


def get_actions_for_eval(cfg: DictConfig, dataset) -> Optional[Sequence[str]]:
    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is not None and hasattr(eval_cfg, "actions"):
        return list(eval_cfg.actions)
    target = str(getattr(cfg.data, "_target_", "")).lower()
    module = dataset.__class__.__module__.lower()
    if "h36motion3d" in target or "h36motion3d" in module:
        return H36M_ACTIONS
    return None


def save_csv_results(rows: List[List[object]], header: List[str], output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return csv_path


def eval_loader(
    cfg: DictConfig,
    dataloader: DataLoader,
    dataset,
    device: torch.device,
    rf_model,
    encoder,
    decoder,
    max_vis: int,
) -> Dict[str, object]:
    eval_cfg = getattr(cfg, "eval", None)
    max_batches = getattr(eval_cfg, "max_batches", None) if eval_cfg is not None else None
    scale = infer_eval_scale(cfg, dataset)

    obs_len = cfg.data.opt.input_n
    pred_len = cfg.data.opt.output_n
    diffusion_steps = getattr(cfg, "diffusion_steps", 20)

    expected_joints = None
    if hasattr(cfg, "in_features"):
        expected_joints = int(cfg.in_features // 3)
    elif hasattr(cfg, "pose_dim"):
        expected_joints = int(cfg.pose_dim // 3)

    dims_to_use = None
    if expected_joints is not None and hasattr(dataset, "dimensions_to_use"):
        dims_candidate = np.asarray(dataset.dimensions_to_use)
        if dims_candidate.size == expected_joints * 3:
            dims_to_use = dims_candidate

    joint_used = None
    if expected_joints is not None and dims_to_use is None and hasattr(dataset, "joint_used"):
        joints_candidate = np.asarray(dataset.joint_used)
        if joints_candidate.size == expected_joints:
            joint_used = joints_candidate

    mse_meter = AverageMeter()
    vis_samples: List[Dict[str, torch.Tensor]] = []
    mpjpe_sum = torch.zeros(pred_len, dtype=torch.float64)
    n_samples = 0

    for batch_idx, joints in enumerate(tqdm(dataloader)):
        if isinstance(joints, (list, tuple)):
            joints = joints[0]

        poses = prepare_pose_sequence(joints, device, dims_to_use=dims_to_use)
        if joint_used is not None and poses.shape[2] != expected_joints:
            max_joint = int(np.max(joint_used))
            if poses.shape[2] > max_joint:
                joints = torch.as_tensor(joint_used, device=poses.device)
                poses = torch.index_select(poses, 2, joints)

        B, T, _, _ = poses.shape
        poses_obs = poses[:, :obs_len, :, :]
        poses_gt_future = poses[:, obs_len:, :, :]

        pred_pose = sample_future_pose(
            rf_model,
            encoder,
            decoder,
            poses_obs,
            pred_len,
            num_steps=diffusion_steps,
        )

        mse = masked_mse(pred_pose, poses_gt_future, mask=torch.isfinite(poses_gt_future))
        mse_meter.update(mse.item(), B)

        pred_eval = pred_pose * scale
        gt_eval = poses_gt_future * scale
        mpjpe_sum += compute_mpjpe_by_timestep(pred_eval, gt_eval).cpu()
        n_samples += B

        if len(vis_samples) < max_vis:
            budget = max_vis - len(vis_samples)
            vis_samples.extend(
                prepare_visual_samples(
                    poses_obs,
                    pred_pose,
                    poses_gt_future,
                    budget,
                )
            )

        if max_batches is not None and (batch_idx + 1) >= int(max_batches):
            break

    if n_samples == 0:
        return {"ret": {}, "mse": mse_meter.avg, "vis_samples": vis_samples}

    mpjpe_avg = mpjpe_sum / n_samples
    ret = {f"#{i + 1}": float(mpjpe_avg[i]) for i in range(pred_len)}
    return {"ret": ret, "mse": mse_meter.avg, "vis_samples": vis_samples}


@torch.no_grad()
def run_eval(cfg: DictConfig, logger):
    device = torch.device(cfg.device)
    use_autoencoder = getattr(cfg, "use_pose_autoencoder", True)
    rf_model = load_motion_model(cfg, device, logger)
    if (not use_autoencoder) and hasattr(rf_model, "latent_dim") and hasattr(cfg, "pose_dim"):
        if rf_model.latent_dim != cfg.pose_dim:
            raise ValueError(
                f"MotionDiffusion latent_dim ({rf_model.latent_dim}) must match pose_dim ({cfg.pose_dim}) "
                "when running without the pose autoencoder."
            )
    encoder, decoder = load_encoder_decoder(cfg, device, logger)

    eval_cfg = getattr(cfg, "eval", None)
    max_vis = int(getattr(eval_cfg, "num_visualizations", 0)) if eval_cfg is not None else 0
    output_dir = "eval_results"
    if eval_cfg is not None and hasattr(eval_cfg, "output_dir"):
        output_dir = eval_cfg.output_dir
    csv_name = "eval_by_action.csv"
    if eval_cfg is not None and hasattr(eval_cfg, "csv_name"):
        csv_name = eval_cfg.csv_name

    dataset_probe = instantiate(cfg.data, split=2)
    actions = get_actions_for_eval(cfg, dataset_probe)
    rows: List[List[object]] = []
    pred_len = cfg.data.opt.output_n
    header = ["action"] + [f"#{i + 1}" for i in range(pred_len)]

    if actions is None:
        dataloader = DataLoader(
            dataset_probe,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        logger.info("Evaluating on %d annotations (%d batches).", len(dataloader.dataset), len(dataloader))
        result = eval_loader(cfg, dataloader, dataset_probe, device, rf_model, encoder, decoder, max_vis)
        ret = result["ret"]
        if not ret:
            logger.warning("No samples processed during evaluation.")
            return {}
        logger.info("Masked MSE: %.4f", result["mse"])
        logger.info("MPJPE per timestep: %s", ret)
        rows.append(["all"] + [ret[f"#{i + 1}"] for i in range(pred_len)])
        save_visualizations(result["vis_samples"], cfg, logger, infer_eval_scale(cfg, dataset_probe))
    else:
        logger.info("Evaluating per action (%d actions).", len(actions))
        per_action_vals = []
        for idx, act in enumerate(actions):
            dataset_act = instantiate(cfg.data, split=2, actions=[act])
            dataloader_act = DataLoader(
                dataset_act,
                batch_size=cfg.test_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
            )
            logger.info("Action %s: %d annotations (%d batches).", act, len(dataloader_act.dataset), len(dataloader_act))
            vis_budget = max_vis
            result = eval_loader(cfg, dataloader_act, dataset_act, device, rf_model, encoder, decoder, vis_budget)
            ret = result["ret"]
            if not ret:
                logger.warning("No samples processed for action %s.", act)
                continue
            logger.info("Action %s | Masked MSE: %.4f", act, result["mse"])
            logger.info("Action %s | MPJPE per timestep: %s", act, ret)
            rows.append([act] + [ret[f"#{i + 1}"] for i in range(pred_len)])
            per_action_vals.append([ret[f"#{i + 1}"] for i in range(pred_len)])
            save_visualizations(
                result["vis_samples"],
                cfg,
                logger,
                infer_eval_scale(cfg, dataset_act),
                subdir=str(act),
            )

        if per_action_vals:
            avg_vals = np.mean(np.asarray(per_action_vals, dtype=float), axis=0).tolist()
            rows.append(["average"] + avg_vals)

    csv_path = save_csv_results(rows, header, os.path.join(os.getcwd(), output_dir), csv_name)
    logger.info("Saved evaluation CSV to %s", csv_path)
    return {row[0]: row[1:] for row in rows}


@hydra.main(config_path="config", config_name="configH36m")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = create_logger("")
    logger.info("Evaluation config:")
    logger.info(cfg)

    run_eval(cfg, logger)


if __name__ == "__main__":
    main()
