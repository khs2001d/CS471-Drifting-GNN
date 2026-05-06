# -*- coding: utf-8 -*-
import argparse
import json
import os

import torch

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.checkpoint import load_teacher_checkpoint
from algorithm.diffstg.model import DiffSTG
from eval_teacher_pems08 import apply_checkpoint_config
from train import default_config, setup_seed
from utils.common_utils import to_device


def tensor_stats(x):
    x_float = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(x_float.mean().item()),
        "std": float(x_float.std().item()),
        "min": float(x_float.min().item()),
        "max": float(x_float.max().item()),
    }


def build_loader(clean_data, config, split, batch_size):
    if split == "train":
        data_range = (0 + config.model.T_p, config.data.val_start_idx - config.model.T_p + 1)
    elif split == "val":
        data_range = (config.data.val_start_idx + config.model.T_p, config.data.test_start_idx - config.model.T_p + 1)
    else:
        raise ValueError(f"Unsupported split: {split}")
    dataset = TrafficDataset(clean_data, data_range, config)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    return dataset, loader


def save_metadata(output_root, args, config, checkpoint, clean_data):
    os.makedirs(output_root, exist_ok=True)
    metadata = {
        "dataset": "PEMS08",
        "checkpoint": args.checkpoint,
        "sample_strategy": "ddim_multi",
        "sample_steps": args.sample_steps,
        "num_teacher_samples": args.num_teacher_samples,
        "cache_scale": "normalized",
        "shape_convention": {
            "history": "(B, T_h, V, F), normalized",
            "gt_future": "(B, T_p, V, F), normalized",
            "x_masked": "(B, F, V, T_h + T_p), normalized, future portion zero",
            "teacher_samples": "(B, S_T, T_p, V, F), normalized",
            "pos_w": "(B, T_h)",
            "pos_d": "(B, T_h)",
        },
        "normalization": {
            "mean": float(clean_data.mean),
            "std": float(clean_data.std),
            "note": "All cached traffic tensors are normalized with this scaler. Reverse-normalize only for reporting metrics.",
        },
        "model": {
            "N": int(config.model.N),
            "T_h": int(config.model.T_h),
            "T_p": int(config.model.T_p),
            "V": int(config.model.V),
            "F": int(config.model.F),
            "epsilon_theta": config.model.epsilon_theta,
        },
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_validation_metric": checkpoint.get("best_validation_metric", {}),
    }
    with open(os.path.join(output_root, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def generate_split_cache(model, clean_data, config, args, split):
    dataset, loader = build_loader(clean_data, config, split, args.batch_size)
    split_dir = os.path.join(args.output_root, split)
    os.makedirs(split_dir, exist_ok=True)

    shard_paths = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            future, history, pos_w, pos_d = to_device(batch, config.device)
            x_masked_bt = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
            x_masked = x_masked_bt.transpose(1, 3)

            x_hat = model((x_masked, pos_w, pos_d), args.num_teacher_samples)
            teacher_future = x_hat[:, :, :, :, -config.model.T_p:].transpose(2, 4).contiguous()

            shard = {
                "split": split,
                "batch_idx": batch_idx,
                "indices": torch.arange(
                    batch_idx * args.batch_size,
                    batch_idx * args.batch_size + future.shape[0],
                    dtype=torch.long,
                ),
                "history": history.detach().cpu().contiguous(),
                "x_masked": x_masked.detach().cpu().contiguous(),
                "gt_future": future.detach().cpu().contiguous(),
                "teacher_samples": teacher_future.detach().cpu().contiguous(),
                "pos_w": pos_w.detach().cpu().contiguous(),
                "pos_d": pos_d.detach().cpu().contiguous(),
                "adj": torch.from_numpy(clean_data.adj).float(),
                "cache_scale": "normalized",
                "normalization": {
                    "mean": float(clean_data.mean),
                    "std": float(clean_data.std),
                },
                "shape_notes": {
                    "history": "(B, T_h, V, F)",
                    "x_masked": "(B, F, V, T_h + T_p)",
                    "gt_future": "(B, T_p, V, F)",
                    "teacher_samples": "(B, S_T, T_p, V, F)",
                },
            }
            shard_path = os.path.join(split_dir, f"shard_{batch_idx:05d}.pt")
            torch.save(shard, shard_path)
            shard_paths.append(shard_path)
            print(
                f"saved {shard_path} "
                f"teacher_samples={tuple(shard['teacher_samples'].shape)} "
                f"gt_future={tuple(shard['gt_future'].shape)}"
            )

    index = {
        "split": split,
        "num_dataset_examples": len(dataset),
        "num_shards": len(shard_paths),
        "shards": shard_paths,
        "cache_scale": "normalized",
    }
    with open(os.path.join(split_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def load_teacher(args):
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    checkpoint = load_teacher_checkpoint(args.checkpoint, map_location="cpu")
    config = default_config("PEMS08")
    config = apply_checkpoint_config(config, checkpoint)
    config.model.sample_steps = args.sample_steps
    config.model.sample_strategy = "ddim_multi"
    config.n_samples = args.num_teacher_samples
    config.batch_size = args.batch_size
    config.is_test = False

    clean_data = CleanDataset(config)
    scaler = checkpoint.get("scaler", {})
    if "mean" in scaler and "std" in scaler:
        clean_data.mean = float(scaler["mean"])
        clean_data.std = float(scaler["std"])

    config.model.A = clean_data.adj
    model = DiffSTG(config.model).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.set_sample_strategy("ddim_multi")
    model.set_ddim_sample_steps(args.sample_steps)
    model.eval()
    return model, clean_data, config, checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cached DiffSTG teacher samples for PEMS08.")
    parser.add_argument("--checkpoint", type=str, default="outputs/pems08_teacher/checkpoints/best.pt")
    parser.add_argument("--output_root", type=str, default="outputs/pems08_teacher_cache")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--num_teacher_samples", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num_threads", type=int, default=2)
    return parser.parse_args()


def main(args):
    model, clean_data, config, checkpoint = load_teacher(args)
    save_metadata(args.output_root, args, config, checkpoint, clean_data)
    for split in args.splits:
        generate_split_cache(model, clean_data, config, args, split)


if __name__ == "__main__":
    main(parse_args())
