# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from timeit import default_timer as timer

import numpy as np
import torch

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.checkpoint import load_teacher_checkpoint
from algorithm.diffstg.model import DiffSTG
from train import default_config, setup_seed
from utils.common_utils import to_device


def sample_crps_np(samples: np.ndarray, target: np.ndarray) -> float:
    """
    Empirical sample CRPS.
    samples: (B, S, ...)
    target:  (B, ...)
    """
    abs_error = np.mean(np.abs(samples - target[:, None, ...]))
    pairwise = np.mean(np.abs(samples[:, :, None, ...] - samples[:, None, :, ...]))
    return float(abs_error - 0.5 * pairwise)


def sample_diversity_np(samples: np.ndarray) -> float:
    """
    Mean pairwise absolute difference between generated samples.
    samples: (B, S, ...)
    """
    if samples.shape[1] < 2:
        return 0.0
    pairwise = np.abs(samples[:, :, None, ...] - samples[:, None, :, ...])
    return float(pairwise.mean())


def masked_mae_np(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.0) -> float:
    mask = np.not_equal(y_true, null_val).astype("float32")
    mask /= mask.mean()
    return float(np.mean(np.nan_to_num(mask * np.abs(y_true - y_pred))))


def masked_rmse_np(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.0) -> float:
    mask = np.not_equal(y_true, null_val).astype("float32")
    mask /= mask.mean()
    mse = np.mean(np.nan_to_num(mask * ((y_true - y_pred) ** 2)))
    return float(mse ** 0.5)


def apply_checkpoint_config(config, checkpoint):
    ckpt_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})

    config.model.N = int(ckpt_args.get("N", model_config.get("N", config.model.N)))
    config.model.T_h = int(ckpt_args.get("T_h", model_config.get("T_h", config.model.T_h)))
    config.model.T_p = int(ckpt_args.get("T_p") or model_config.get("T_p", config.model.T_p))
    config.T_h = config.model.T_h
    config.T_p = config.model.T_p
    config.model.epsilon_theta = ckpt_args.get("epsilon_theta", model_config.get("epsilon_theta", config.model.epsilon_theta))
    config.model.d_h = int(ckpt_args.get("hidden_size", model_config.get("d_h", config.model.d_h)))
    config.model.C = config.model.d_h
    config.model.n_channels = config.model.d_h
    config.model.beta_end = float(ckpt_args.get("beta_end", model_config.get("beta_end", config.model.beta_end)))
    config.model.beta_schedule = ckpt_args.get("beta_schedule", model_config.get("beta_schedule", config.model.beta_schedule))
    config.model.sample_strategy = "ddim_multi"
    return config


def evaluate(args):
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    checkpoint = load_teacher_checkpoint(args.checkpoint, map_location="cpu")
    config = default_config("PEMS08")
    config = apply_checkpoint_config(config, checkpoint)
    config.model.sample_steps = args.sample_steps
    config.n_samples = args.num_samples
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

    test_dataset = TrafficDataset(
        clean_data,
        (config.data.test_start_idx + config.model.T_p, -1),
        config,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False)

    y_true_all, y_pred_all = [], []
    inference_times = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if args.max_eval_batches is not None and i >= args.max_eval_batches:
                break
            future, history, pos_w, pos_d = to_device(batch, config.device)
            x = torch.cat((history, future), dim=1).to(config.device)
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
            x = x.transpose(1, 3)
            x_masked = x_masked.transpose(1, 3)

            if config.device.type == "cuda":
                torch.cuda.synchronize(config.device)
            start = timer()
            x_hat = model((x_masked, pos_w, pos_d), args.num_samples)
            if config.device.type == "cuda":
                torch.cuda.synchronize(config.device)
            inference_times.append(timer() - start)

            x = clean_data.reverse_normalization(x)
            x_hat = clean_data.reverse_normalization(x_hat).detach()
            y_true = x[:, :, :, -config.model.T_p:].transpose(1, 3).cpu().numpy()
            y_pred = x_hat[:, :, :, :, -config.model.T_p:].transpose(2, 4).cpu().numpy()
            y_pred = np.clip(y_pred, 0, np.inf)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    y_mean = np.mean(y_pred, axis=1)

    results = {
        "model": "DiffSTG_teacher",
        "dataset": "PEMS08",
        "checkpoint": args.checkpoint,
        "sample_strategy": "ddim_multi",
        "sample_steps": args.sample_steps,
        "num_samples": args.num_samples,
        "NFE": args.sample_steps,
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
    stem = f"teacher_pems08_steps{args.sample_steps}_samples{args.num_samples}"
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
    parser = argparse.ArgumentParser(description="Evaluate a trained DiffSTG teacher on PEMS08.")
    parser.add_argument("--checkpoint", type=str, default="outputs/pems08_teacher/checkpoints/best.pt")
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="outputs/pems08_results")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num_threads", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
