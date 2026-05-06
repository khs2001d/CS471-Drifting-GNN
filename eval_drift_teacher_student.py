# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from timeit import default_timer as timer

import numpy as np
import torch

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.drifting.student import DriftTeacherStudent
from eval_teacher_pems08 import masked_mae_np, masked_rmse_np, sample_crps_np
from train import default_config, setup_seed
from utils.common_utils import to_device


def sample_diversity_np(samples):
    if samples.shape[1] < 2:
        return 0.0
    pairwise = np.abs(samples[:, :, None, ...] - samples[:, None, :, ...])
    return float(pairwise.mean())


def build_config_from_checkpoint(checkpoint, device):
    config = default_config("PEMS08")
    model_config = checkpoint["model_config"]
    config.model.T_h = int(model_config["T_h"])
    config.model.T_p = int(model_config["T_p"])
    config.T_h = config.model.T_h
    config.T_p = config.model.T_p
    config.model.V = int(model_config["V"])
    config.model.F = int(model_config["F"])
    config.model.d_h = int(model_config["d_h"])
    config.model.C = config.model.d_h
    config.model.n_channels = config.model.d_h
    config.model.channel_multipliers = list(model_config["channel_multipliers"])
    config.model.sample_steps = 1
    config.model.sample_strategy = "one_step_student"
    config.model.device = device
    config.device = device
    return config


def evaluate(args):
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = build_config_from_checkpoint(checkpoint, device)

    clean_data = CleanDataset(config)
    norm = checkpoint.get("normalization", {})
    if "mean" in norm and "std" in norm:
        clean_data.mean = float(norm["mean"])
        clean_data.std = float(norm["std"])
    config.model.A = clean_data.adj

    model = DriftTeacherStudent(config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False)

    y_true_all, y_pred_all = [], []
    inference_times = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if args.max_eval_batches is not None and i >= args.max_eval_batches:
                break
            future, history, pos_w, pos_d = to_device(batch, device)
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).transpose(1, 3).contiguous()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = timer()
            y_pred = model(x_masked, pos_w, pos_d, args.num_samples)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_times.append(timer() - start)

            y_true = clean_data.reverse_normalization(future).cpu().numpy()
            y_pred = clean_data.reverse_normalization(y_pred).detach().cpu().numpy()
            y_pred = np.clip(y_pred, 0, np.inf)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    y_mean = y_pred.mean(axis=1)

    results = {
        "model": "DriftTeacherStudent",
        "dataset": "PEMS08",
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "NFE": 1,
        "MAE": masked_mae_np(y_true, y_mean, 0.0),
        "RMSE": masked_rmse_np(y_true, y_mean, 0.0),
        "CRPS": sample_crps_np(y_pred, y_true),
        "sample_diversity": sample_diversity_np(y_pred),
        "inference_time_sec": float(np.sum(inference_times)),
        "num_eval_batches": len(inference_times),
        "num_eval_examples": int(y_true.shape[0]),
        "target_shape": list(y_true.shape),
        "sample_shape": list(y_pred.shape),
    }
    if results["num_eval_examples"] > 0:
        results["time_per_example_sec"] = results["inference_time_sec"] / results["num_eval_examples"]

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"student_drift_teacher_samples{args.num_samples}"
    if args.max_eval_batches is not None:
        stem += f"_smoke{args.max_eval_batches}"
    json_path = os.path.join(args.output_dir, stem + ".json")
    csv_path = os.path.join(args.output_dir, stem + ".csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)

    print(json.dumps(results, indent=2))
    print(f"saved_json: {json_path}")
    print(f"saved_csv: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one-step Drift-to-teacher student on PEMS08.")
    parser.add_argument("--checkpoint", type=str, default="outputs/pems08_student_drift_teacher/checkpoints/best.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/pems08_student_drift_teacher/results")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num_threads", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
