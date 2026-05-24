# -*- coding: utf-8 -*-
import argparse
import json
import os
from timeit import default_timer as timer

import torch

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.checkpoint import load_teacher_checkpoint
from algorithm.diffstg.model import DiffSTG
from algorithm.drifting.student import DriftTeacherStudent
from eval_drift_teacher_student import build_config_from_checkpoint as build_student_config_from_checkpoint
from eval_teacher_pems08 import apply_checkpoint_config
from train import default_config, setup_seed
from utils.common_utils import to_device


def _state_prefixes(state_dict, n=8):
    return list(state_dict.keys())[:n]


def validate_checkpoint_kind(checkpoint, model_type, checkpoint_path):
    state = checkpoint.get("model_state_dict", {})
    keys = _state_prefixes(state)
    if model_type == "teacher":
        if not any(k.startswith("eps_model.") for k in state.keys()):
            raise RuntimeError(
                "Checkpoint does not look like a DiffSTG teacher checkpoint. "
                f"Expected state_dict keys starting with 'eps_model.', got first keys={keys}. "
                f"checkpoint={checkpoint_path}"
            )
    elif model_type == "student":
        if not any(k.startswith("backbone.") for k in state.keys()):
            raise RuntimeError(
                "Checkpoint does not look like a DriftTeacherStudent checkpoint. "
                f"Expected state_dict keys starting with 'backbone.', got first keys={keys}. "
                f"checkpoint={checkpoint_path}"
            )


def load_teacher(args, device):
    checkpoint = load_teacher_checkpoint(args.checkpoint, map_location="cpu")
    validate_checkpoint_kind(checkpoint, "teacher", args.checkpoint)
    config = default_config("PEMS08")
    config = apply_checkpoint_config(config, checkpoint)
    config.model.sample_steps = args.sample_steps
    config.n_samples = args.num_samples
    config.batch_size = args.batch_size
    config.is_test = False
    config.device = device
    config.model.device = device

    clean_data = CleanDataset(config)
    scaler = checkpoint.get("scaler", {})
    if "mean" in scaler and "std" in scaler:
        clean_data.mean = float(scaler["mean"])
        clean_data.std = float(scaler["std"])
    config.model.A = clean_data.adj

    model = DiffSTG(config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.set_sample_strategy("ddim_multi")
    model.set_ddim_sample_steps(args.sample_steps)
    model.eval()
    return model, clean_data, config, checkpoint


def load_student(args, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    validate_checkpoint_kind(checkpoint, "student", args.checkpoint)
    config = build_student_config_from_checkpoint(checkpoint, device)
    clean_data = CleanDataset(config)
    norm = checkpoint.get("normalization", {})
    if "mean" in norm and "std" in norm:
        clean_data.mean = float(norm["mean"])
        clean_data.std = float(norm["std"])
    config.model.A = clean_data.adj

    model = DriftTeacherStudent(config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, clean_data, config, checkpoint


def build_test_loader(clean_data, config, batch_size):
    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataset, test_loader


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def generate(args):
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "teacher":
        model, clean_data, config, checkpoint = load_teacher(args, device)
        nfe = int(args.sample_steps)
    else:
        model, clean_data, config, checkpoint = load_student(args, device)
        nfe = 1

    dataset, loader = build_test_loader(clean_data, config, args.batch_size)
    os.makedirs(args.output_dir, exist_ok=True)
    shard_paths = []
    inference_times = []
    total_examples = 0

    print(f"model_type={args.model_type}")
    print(f"checkpoint={args.checkpoint}")
    print(f"output_dir={args.output_dir}")
    print(f"num_samples={args.num_samples} sample_steps={args.sample_steps} NFE={nfe}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
                break
            future, history, pos_w, pos_d = to_device(batch, device)
            x_masked_bt = torch.cat((history, torch.zeros_like(future)), dim=1).to(device)
            x_masked = x_masked_bt.transpose(1, 3).contiguous()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = timer()
            if args.model_type == "teacher":
                x_hat = model((x_masked, pos_w, pos_d), args.num_samples)
                y_samples = x_hat[:, :, :, :, -config.model.T_p:].transpose(2, 4).contiguous()
            else:
                y_samples = model(x_masked, pos_w, pos_d, args.num_samples)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = timer() - start
            inference_times.append(elapsed)

            y_true = clean_data.reverse_normalization(future).detach().cpu().contiguous()
            y_samples = clean_data.reverse_normalization(y_samples).detach().cpu().contiguous().clamp_min(0.0)
            history_orig = clean_data.reverse_normalization(history).detach().cpu().contiguous()

            bsz = int(future.shape[0])
            indices = torch.arange(total_examples, total_examples + bsz, dtype=torch.long)
            data_indices = torch.arange(
                config.data.test_start_idx + config.model.T_p + total_examples,
                config.data.test_start_idx + config.model.T_p + total_examples + bsz,
                dtype=torch.long,
            )
            total_examples += bsz

            shard = {
                "indices": indices,
                "data_indices": data_indices,
                "history": history_orig,
                "gt_future": y_true,
                "samples": y_samples,
                "inference_time_sec": float(elapsed),
                "batch_idx": int(batch_idx),
                "model_type": args.model_type,
                "sample_scale": "original",
                "normalization": {"mean": float(clean_data.mean), "std": float(clean_data.std)},
            }
            if args.model_type == "teacher":
                shard["teacher_samples"] = y_samples
            else:
                shard["student_samples"] = y_samples
            shard_path = os.path.join(args.output_dir, f"shard_{batch_idx:05d}.pt")
            torch.save(shard, shard_path)
            shard_paths.append(shard_path)
            print(
                f"saved {shard_path} indices=({int(indices[0])},{int(indices[-1])}) "
                f"samples={tuple(y_samples.shape)} gt={tuple(y_true.shape)} time={elapsed:.4f}s"
            )

    metadata = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "dataset": "PEMS08",
        "split": "test",
        "sample_steps": int(args.sample_steps),
        "num_samples": int(args.num_samples),
        "NFE": int(nfe),
        "num_eval_examples": int(total_examples),
        "num_eval_batches": len(shard_paths),
        "inference_time_total_sec": float(sum(inference_times)),
        "inference_time_per_batch_sec": float(sum(inference_times) / max(len(inference_times), 1)),
        "sample_shape_note": "samples: (B, S, T_p, V, F), original scale",
        "target_shape_note": "gt_future: (B, T_p, V, F), original scale",
        "normalization": {"mean": float(clean_data.mean), "std": float(clean_data.std)},
        "shards": shard_paths,
    }
    save_json(os.path.join(args.output_dir, "index.json"), metadata)
    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fresh PEMS08 test samples for distribution comparison.")
    parser.add_argument("--model_type", choices=["teacher", "student"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_steps", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num_threads", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
