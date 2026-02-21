from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Optional, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import data_utils
from utils import forward_kinematics as fk_utils
from utils.utils import create_logger

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


def expmap_to_xyz_seq(expmap_seq: torch.Tensor, use_clean_fk: bool = False) -> torch.Tensor:
    b, t, j, _ = expmap_seq.shape
    exp = torch.zeros(b * t, 3 + j * 3, device=expmap_seq.device, dtype=expmap_seq.dtype)
    exp[:, 3:] = expmap_seq.reshape(b * t, j * 3)
    xyz = clean_fk_from_expmap(exp) if use_clean_fk else data_utils.expmap2xyz_torch(exp)
    return xyz.view(b, t, j, 3)


def parse_batch(batch, num_joints):
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


def get_actions_for_eval(cfg: DictConfig, dataset) -> Optional[Sequence[str]]:
    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is not None and hasattr(eval_cfg, "actions"):
        return list(eval_cfg.actions)
    target = str(getattr(cfg.data, "_target_", "")).lower()
    module = dataset.__class__.__module__.lower()
    if "h36motion3d" in target or "h36motion3d" in module or "h36motion" in module:
        return H36M_ACTIONS
    return None


def prepare_input_rep(
    rot_expmap: torch.Tensor,
    input_representation: str,
    use_clean_fk: bool,
    data_scale: float,
) -> torch.Tensor:
    if data_scale <= 0:
        raise ValueError(f"data_scale must be > 0, got {data_scale}")
    if input_representation == "expmap":
        return rot_expmap
    if input_representation == "xyz":
        return expmap_to_xyz_seq(rot_expmap, use_clean_fk=use_clean_fk) / data_scale
    raise ValueError(f"Unsupported input_representation: {input_representation}")


def rep_to_xyz(
    rep: torch.Tensor,
    input_representation: str,
    use_clean_fk: bool,
) -> torch.Tensor:
    if input_representation == "xyz":
        return rep
    if input_representation == "expmap":
        return expmap_to_xyz_seq(rep, use_clean_fk=use_clean_fk)
    raise ValueError(f"Unsupported input_representation: {input_representation}")


def unscale_xyz_for_metric(
    xyz_in_model_scale: torch.Tensor,
    input_representation: str,
    data_scale: float,
) -> torch.Tensor:
    if input_representation == "xyz":
        return xyz_in_model_scale * data_scale
    return xyz_in_model_scale


def save_csv_results(rows: List[List[object]], header: List[str], output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return csv_path


def load_model(cfg: DictConfig, device: torch.device, logger):
    model = instantiate(cfg.encoderModel).to(device)
    eval_cfg = getattr(cfg, "eval", None)
    ckpt_path = getattr(eval_cfg, "checkpoint", None) if eval_cfg is not None else None
    if ckpt_path is None:
        raise ValueError("Set eval.checkpoint in config for AuxFormer NEPA Joint evaluation.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        "Loaded AuxFormer NEPA Joint checkpoint from %s (epoch=%s)",
        ckpt_path,
        ckpt.get("epoch", "n/a"),
    )
    return model


def eval_loader(
    cfg: DictConfig,
    dataloader: DataLoader,
    dataset,
    device: torch.device,
    model,
) -> Dict[str, object]:
    eval_cfg = getattr(cfg, "eval", None)
    max_batches = getattr(eval_cfg, "max_batches", None) if eval_cfg is not None else None
    scale = infer_eval_scale(cfg, dataset)
    use_clean_fk = cfg.fk.mode == "clean"
    input_representation = str(cfg.auxformer_nepa.input_representation).lower()
    data_scale = float(getattr(cfg.auxformer_nepa, "data_scale", 1.0))
    infer_use_denoised_feedback = bool(cfg.auxformer_nepa.infer_use_denoised_feedback)

    if data_scale <= 0:
        raise ValueError(f"auxformer_nepa.data_scale must be > 0, got {data_scale}")

    obs_len = int(cfg.data.opt.input_n)
    pred_len = int(cfg.data.opt.output_n)
    total_len = obs_len + pred_len

    mpjpe_sum = torch.zeros(pred_len, dtype=torch.float64)
    mae_xyz_sum = torch.zeros(pred_len, dtype=torch.float64)
    n_samples = 0

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        rot_gt = parse_batch(batch, cfg.num_joints).to(device)
        rot_gt = rot_gt[:, :total_len]

        pose_in = prepare_input_rep(
            rot_gt,
            input_representation=input_representation,
            use_clean_fk=use_clean_fk,
            data_scale=data_scale,
        )
        pred_future_rep = model.predict_rollout(
            pose_in[:, :obs_len],
            future_frames=pred_len,
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
        gt_future_xyz = expmap_to_xyz_seq(
            rot_gt[:, obs_len : obs_len + pred_len],
            use_clean_fk=use_clean_fk,
        )

        mae_t = torch.mean(torch.abs(pred_future_xyz - gt_future_xyz), dim=(0, 2, 3))
        mae_xyz_sum += mae_t.detach().cpu().double() * pred_future_xyz.size(0)

        pred_future_m = pred_future_xyz * scale
        gt_future_m = gt_future_xyz * scale
        mpjpe_t = torch.norm(pred_future_m - gt_future_m, dim=-1).mean(dim=2)
        mpjpe_sum += mpjpe_t.detach().cpu().double().sum(dim=0)
        n_samples += pred_future_xyz.size(0)

    if n_samples == 0:
        return {"mpjpe": {}, "mae_xyz": {}}

    mpjpe_avg = (mpjpe_sum / n_samples).tolist()
    mae_xyz_avg = (mae_xyz_sum / n_samples).tolist()
    mpjpe = {f"#{i + 1}": float(v) for i, v in enumerate(mpjpe_avg)}
    mae_xyz = {f"#{i + 1}": float(v) for i, v in enumerate(mae_xyz_avg)}
    return {"mpjpe": mpjpe, "mae_xyz": mae_xyz}


def run_eval(cfg: DictConfig, logger):
    dataset = instantiate(cfg.data, split=2)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    device = torch.device(cfg.device)
    model = load_model(cfg, device, logger)

    output_dir = cfg.eval.output_dir
    csv_name = cfg.eval.csv_name
    pred_len = int(cfg.data.opt.output_n)
    header = ["action", "metric"] + [f"t{i + 1}" for i in range(pred_len)]
    rows = []

    actions = get_actions_for_eval(cfg, dataset)
    if actions is None:
        result = eval_loader(cfg, dataloader, dataset, device, model)
        mpjpe = result["mpjpe"]
        mae_xyz = result["mae_xyz"]
        if mpjpe:
            rows.append(["all", "MPJPE"] + [mpjpe[f"#{i + 1}"] for i in range(pred_len)])
        if mae_xyz:
            rows.append(["all", "MAE_XYZ"] + [mae_xyz[f"#{i + 1}"] for i in range(pred_len)])
    else:
        per_action_mpjpe = []
        per_action_mae_xyz = []
        for act in actions:
            dataset_act = instantiate(cfg.data, split=2, actions=[act])
            dataloader_act = DataLoader(
                dataset_act,
                batch_size=cfg.test_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
            )
            result = eval_loader(cfg, dataloader_act, dataset_act, device, model)
            mpjpe = result["mpjpe"]
            mae_xyz = result["mae_xyz"]
            if not mpjpe or not mae_xyz:
                logger.warning("No samples processed for action %s.", act)
                continue
            logger.info("Action %s | MPJPE per timestep: %s", act, mpjpe)
            logger.info("Action %s | MAE_XYZ per timestep: %s", act, mae_xyz)
            rows.append([act, "MPJPE"] + [mpjpe[f"#{i + 1}"] for i in range(pred_len)])
            rows.append([act, "MAE_XYZ"] + [mae_xyz[f"#{i + 1}"] for i in range(pred_len)])
            per_action_mpjpe.append([mpjpe[f"#{i + 1}"] for i in range(pred_len)])
            per_action_mae_xyz.append([mae_xyz[f"#{i + 1}"] for i in range(pred_len)])

        if per_action_mpjpe:
            avg_mpjpe = np.mean(np.asarray(per_action_mpjpe, dtype=float), axis=0).tolist()
            avg_mae_xyz = np.mean(np.asarray(per_action_mae_xyz, dtype=float), axis=0).tolist()
            rows.append(["average", "MPJPE"] + avg_mpjpe)
            rows.append(["average", "MAE_XYZ"] + avg_mae_xyz)

    csv_path = save_csv_results(rows, header, os.path.join(os.getcwd(), output_dir), csv_name)
    logger.info("Saved evaluation CSV to %s", csv_path)
    return {row[0]: row[2:] for row in rows}


@hydra.main(config_path="config", config_name="configAuxFormer_nepa_joint")
def main(cfg: DictConfig):
    cfg.mode = "eval"

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = create_logger("")
    logger.info("Evaluation config:")
    logger.info(cfg)
    run_eval(cfg, logger)


if __name__ == "__main__":
    main()
