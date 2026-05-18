# -*- coding: utf-8 -*-
import argparse
import json
import os
from bisect import bisect_right
from timeit import default_timer as timer

import numpy as np
import torch

from algorithm.drifting.losses import antisymmetric_drift_to_teacher_loss, rbf_drift_to_teacher_loss
from algorithm.drifting.student import DriftTeacherStudent
from train import default_config, setup_seed


class TeacherCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, split):
        self.split = split
        self.split_dir = os.path.join(cache_dir, split)
        with open(os.path.join(self.split_dir, "index.json"), "r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.shards = self.index["shards"]
        self.sizes = []
        for path in self.shards:
            shard = torch.load(path, map_location="cpu")
            self.sizes.append(int(shard["teacher_samples"].shape[0]))
        self.cumulative = np.cumsum(self.sizes).tolist()
        first = torch.load(self.shards[0], map_location="cpu")
        self.adj = first["adj"].numpy()
        self.normalization = first["normalization"]
        self._last_shard_idx = None
        self._last_shard = None

    def __len__(self):
        return self.cumulative[-1]

    def _load_shard(self, shard_idx):
        if self._last_shard_idx != shard_idx:
            self._last_shard = torch.load(self.shards[shard_idx], map_location="cpu")
            self._last_shard_idx = shard_idx
        return self._last_shard

    def __getitem__(self, idx):
        shard_idx = bisect_right(self.cumulative, idx)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = idx - prev
        shard = self._load_shard(shard_idx)
        return {
            "history": shard["history"][local_idx],
            "x_masked": shard["x_masked"][local_idx],
            "gt_future": shard["gt_future"][local_idx],
            "teacher_samples": shard["teacher_samples"][local_idx],
            "pos_w": shard["pos_w"][local_idx],
            "pos_d": shard["pos_d"][local_idx],
            "index": shard["indices"][local_idx],
        }


def build_student_config(cache_dir, device):
    with open(os.path.join(cache_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    config = default_config("PEMS08")
    model_meta = metadata["model"]
    config.model.N = int(model_meta["N"])
    config.model.T_h = int(model_meta["T_h"])
    config.model.T_p = int(model_meta["T_p"])
    config.model.V = int(model_meta["V"])
    config.model.F = int(model_meta["F"])
    config.model.epsilon_theta = "UGnet"
    config.model.sample_steps = 1
    config.model.sample_strategy = "one_step_student"
    config.model.device = device
    config.device = device
    return config, metadata


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def compute_student_loss(student, teacher_samples, args):
    if args.loss_type == "attraction":
        return rbf_drift_to_teacher_loss(
            student,
            teacher_samples,
            eta=args.eta,
            sigma=args.sigma,
        )
    if args.loss_type == "antisymmetric":
        return antisymmetric_drift_to_teacher_loss(
            student,
            teacher_samples,
            eta=args.eta,
            sigma=args.sigma,
        )
    raise ValueError(f"Unsupported loss_type: {args.loss_type}")


def format_loss_stats(stats):
    fields = [
        ("sigma", ".6f"),
        ("drift_norm", ".6f"),
        ("positive_drift_norm", ".6f"),
        ("negative_drift_norm", ".6f"),
        ("student_diversity", ".6f"),
        ("teacher_diversity", ".6f"),
        ("diversity_ratio", ".6f"),
    ]
    parts = []
    for key, fmt in fields:
        if key in stats:
            parts.append(f"{key}={stats[key]:{fmt}}")
    return " ".join(parts)


def evaluate_loss(model, loader, args, device, max_batches=None):
    model.eval()
    total, n = 0.0, 0
    last_stats = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = to_device(batch, device)
            student = model(batch["x_masked"], batch["pos_w"], batch["pos_d"], args.num_student_samples)
            loss, stats = compute_student_loss(student, batch["teacher_samples"], args)
            total += loss.item()
            n += 1
            last_stats = stats
    model.train()
    return total / max(n, 1), last_stats


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, args, config, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "args": vars(args),
            "model_config": {
                "T_h": int(config.model.T_h),
                "T_p": int(config.model.T_p),
                "V": int(config.model.V),
                "F": int(config.model.F),
                "d_h": int(config.model.d_h),
                "channel_multipliers": list(config.model.channel_multipliers),
            },
            "cache_metadata": metadata,
            "normalization": metadata["normalization"],
            "NFE": 1,
        },
        path,
    )


def train(args):
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TeacherCacheDataset(args.cache_dir, "train")
    val_dataset = TeacherCacheDataset(args.cache_dir, "val")
    config, metadata = build_student_config(args.cache_dir, device)
    config.model.A = train_dataset.adj

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = DriftTeacherStudent(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.output_dir, "checkpoints", "best.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"train_examples={len(train_dataset)} val_examples={len(val_dataset)}")
    print(
        f"NFE=1 loss_type={args.loss_type} num_student_samples={args.num_student_samples} "
        f"eta={args.eta} sigma={args.sigma or 'auto'}"
    )

    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        start = timer()
        for i, batch in enumerate(train_loader):
            if args.max_train_batches is not None and i >= args.max_train_batches:
                break
            batch = to_device(batch, device)
            student = model(batch["x_masked"], batch["pos_w"], batch["pos_d"], args.num_student_samples)
            loss, stats = compute_student_loss(student, batch["teacher_samples"], args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            n += 1
            if i == 0:
                print(
                    "first_batch_shapes "
                    f"x_masked={tuple(batch['x_masked'].shape)} "
                    f"teacher={tuple(batch['teacher_samples'].shape)} "
                    f"student={tuple(student.shape)}"
                )
            print(
                f"epoch={epoch + 1} batch={i + 1} "
                f"loss={loss.item():.6f} {format_loss_stats(stats)}"
            )

        train_loss = running / max(n, 1)
        val_loss, val_stats = evaluate_loss(model, val_loader, args, device, args.max_eval_batches)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} time={timer() - start:.2f}s"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_loss, args, config, metadata)
            print(f"saved_best={ckpt_path} best_val_loss={best_val_loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train one-step Drift-to-teacher student from cached teacher samples.")
    parser.add_argument("--cache_dir", type=str, default="outputs/pems08_teacher_cache")
    parser.add_argument("--output_dir", type=str, default="outputs/pems08_student_drift_teacher")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_student_samples", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--loss_type", type=str, default="attraction", choices=["attraction", "antisymmetric"])
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num_threads", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
