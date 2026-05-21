# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from eval_teacher_pems08 import masked_mae_np, masked_rmse_np, sample_crps_np

COLUMNS = [
    "model",
    "checkpoint",
    "sample_steps",
    "num_samples",
    "NFE",
    "CRPS",
    "MAE",
    "RMSE",
    "sample_diversity",
    "teacher100_diversity",
    "relative_diversity_vs_teacher100",
    "EMD_to_teacher100",
    "EnergyDistance_to_teacher100",
    "inference_time_total_sec",
    "inference_time_per_batch_sec",
    "speedup_vs_teacher100",
    "num_eval_examples",
    "num_eval_batches",
    "output_sample_dir",
    "teacher_reference_dir",
]


def load_index(sample_dir):
    with open(os.path.join(sample_dir, "index.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def get_samples(shard):
    if "samples" in shard:
        return shard["samples"]
    if "teacher_samples" in shard:
        return shard["teacher_samples"]
    return shard["student_samples"]


def offdiag_mean(cost):
    s = cost.shape[0]
    if s <= 1:
        return 0.0
    mask = ~np.eye(s, dtype=bool)
    return float(cost[mask].mean())


def pairwise_l1(a, b):
    # a: (S, ...), b: (K, ...), returns (S, K), per-element L1 ground cost.
    af = a.reshape(a.shape[0], -1).astype(np.float64)
    bf = b.reshape(b.shape[0], -1).astype(np.float64)
    return np.abs(af[:, None, :] - bf[None, :, :]).mean(axis=2)


def condition_distribution_metrics(model_samples, teacher_samples):
    cross = pairwise_l1(model_samples, teacher_samples)
    if model_samples.shape[0] == teacher_samples.shape[0]:
        row, col = linear_sum_assignment(cross)
        emd = float(cross[row, col].mean())
    else:
        raise ValueError(f"EMD requires equal sample counts, got {model_samples.shape[0]} and {teacher_samples.shape[0]}")
    mm = pairwise_l1(model_samples, model_samples)
    tt = pairwise_l1(teacher_samples, teacher_samples)
    model_div = offdiag_mean(mm)
    teacher_div = offdiag_mean(tt)
    energy = float(2.0 * cross.mean() - model_div - teacher_div)
    return emd, energy, model_div, teacher_div


def update_gt_acc(acc, samples, target):
    y_pred = samples
    y_true = target
    y_mean = y_pred.mean(axis=1)
    acc["crps_sum"] += sample_crps_np(y_pred, y_true) * y_true.shape[0]
    acc["mae_sum"] += masked_mae_np(y_true, y_mean, 0.0) * y_true.shape[0]
    acc["rmse_sq_sum"] += (masked_rmse_np(y_true, y_mean, 0.0) ** 2) * y_true.shape[0]
    acc["n_examples"] += y_true.shape[0]


def evaluate_model(model_dir, teacher_dir, max_eval_batches=None, force_self_reference=False):
    model_idx = load_index(model_dir)
    teacher_idx = load_index(teacher_dir)
    model_shards = model_idx["shards"]
    teacher_shards = teacher_idx["shards"]
    if max_eval_batches is not None:
        model_shards = model_shards[:max_eval_batches]
        teacher_shards = teacher_shards[:max_eval_batches]
    if len(model_shards) != len(teacher_shards):
        raise ValueError(f"Shard count mismatch: model={len(model_shards)} teacher={len(teacher_shards)}")

    dist = {"emd": [], "energy": [], "model_div": [], "teacher_div": []}
    gt = {"crps_sum": 0.0, "mae_sum": 0.0, "rmse_sq_sum": 0.0, "n_examples": 0}
    checked = 0

    for shard_i, (mp, tp) in enumerate(zip(model_shards, teacher_shards)):
        ms = torch.load(mp, map_location="cpu")
        ts = torch.load(tp, map_location="cpu")
        mi = ms["indices"].numpy()
        ti = ts["indices"].numpy()
        if not np.array_equal(mi, ti):
            raise ValueError(f"Index mismatch at shard {shard_i}: model {mi[:5]} vs teacher {ti[:5]}")
        m_samples = get_samples(ms).numpy()
        t_samples = get_samples(ts).numpy()
        target = ms["gt_future"].numpy()
        if not np.allclose(target, ts["gt_future"].numpy()):
            raise ValueError(f"GT target mismatch at shard {shard_i}")
        if m_samples.shape[1] != t_samples.shape[1]:
            raise ValueError(f"Sample count mismatch at shard {shard_i}: {m_samples.shape} vs {t_samples.shape}")
        update_gt_acc(gt, m_samples, target)
        for b in range(m_samples.shape[0]):
            if force_self_reference:
                emd, energy = 0.0, 0.0
                div = condition_distribution_metrics(t_samples[b], t_samples[b])[2]
                model_div = div
                teacher_div = div
            else:
                emd, energy, model_div, teacher_div = condition_distribution_metrics(m_samples[b], t_samples[b])
            dist["emd"].append(emd)
            dist["energy"].append(energy)
            dist["model_div"].append(model_div)
            dist["teacher_div"].append(teacher_div)
            checked += 1
        if shard_i == 0:
            print(
                f"index_matching_check=ok shard=0 indices=({mi[0]},{mi[-1]}) "
                f"model_samples={m_samples.shape} teacher_samples={t_samples.shape} gt={target.shape}"
            )

    n = max(gt["n_examples"], 1)
    sample_div = float(np.mean(dist["model_div"]))
    teacher_div = float(np.mean(dist["teacher_div"]))
    total_time = float(model_idx.get("inference_time_total_sec", 0.0))
    if max_eval_batches is not None:
        # Use exact selected shard timings for smoke comparisons.
        total_time = 0.0
        for mp in model_shards:
            total_time += float(torch.load(mp, map_location="cpu").get("inference_time_sec", 0.0))
    num_batches = len(model_shards)
    return {
        "model": model_idx["model_name"],
        "checkpoint": model_idx["checkpoint"],
        "sample_steps": model_idx["sample_steps"],
        "num_samples": model_idx["num_samples"],
        "NFE": model_idx["NFE"],
        "CRPS": float(gt["crps_sum"] / n),
        "MAE": float(gt["mae_sum"] / n),
        "RMSE": float((gt["rmse_sq_sum"] / n) ** 0.5),
        "sample_diversity": sample_div,
        "teacher100_diversity": teacher_div,
        "relative_diversity_vs_teacher100": float(sample_div / teacher_div) if teacher_div > 0 else None,
        "EMD_to_teacher100": float(np.mean(dist["emd"])),
        "EnergyDistance_to_teacher100": float(np.mean(dist["energy"])),
        "inference_time_total_sec": total_time,
        "inference_time_per_batch_sec": float(total_time / max(num_batches, 1)),
        "speedup_vs_teacher100": None,
        "num_eval_examples": int(checked),
        "num_eval_batches": int(num_batches),
        "output_sample_dir": model_dir,
        "teacher_reference_dir": teacher_dir,
    }


def write_results(rows, output_csv, output_json):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main(args):
    specs = [
        ("Teacher100", args.teacher100_dir, True),
        ("Teacher40", args.teacher40_dir, False),
        ("Student_cache8", args.student_cache8_dir, False),
        ("Student_cache32", args.student_cache32_dir, False),
        ("Student_cache32_aux", args.student_cache32_aux_dir, False),
    ]
    rows = []
    for label, sample_dir, self_ref in specs:
        if not sample_dir:
            continue
        print(f"evaluating {label}: {sample_dir}")
        row = evaluate_model(sample_dir, args.teacher100_dir, args.max_eval_batches, force_self_reference=self_ref)
        row["model"] = label
        rows.append(row)
    teacher_time = rows[0]["inference_time_total_sec"] if rows else None
    for row in rows:
        t = row["inference_time_total_sec"]
        row["speedup_vs_teacher100"] = float(teacher_time / t) if teacher_time and t else None
    write_results(rows, args.output_csv, args.output_json)
    print("model,EMD,Energy,Div,RelDiv,CRPS,MAE,RMSE,time,speedup")
    for row in rows:
        print(
            f"{row['model']},{row['EMD_to_teacher100']:.6f},{row['EnergyDistance_to_teacher100']:.6f},"
            f"{row['sample_diversity']:.6f},{row['relative_diversity_vs_teacher100']:.6f},"
            f"{row['CRPS']:.6f},{row['MAE']:.6f},{row['RMSE']:.6f},"
            f"{row['inference_time_total_sec']:.4f},{row['speedup_vs_teacher100']:.4f}"
        )
    print(f"saved_csv={args.output_csv}")
    print(f"saved_json={args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fresh condition-wise distribution preservation metrics on PEMS08.")
    parser.add_argument("--teacher100_dir", type=str, required=True)
    parser.add_argument("--teacher40_dir", type=str, default=None)
    parser.add_argument("--student_cache8_dir", type=str, default=None)
    parser.add_argument("--student_cache32_dir", type=str, default=None)
    parser.add_argument("--student_cache32_aux_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default="outputs/pems08_results/fresh_distribution_comparison_samples32.csv")
    parser.add_argument("--output_json", type=str, default="outputs/pems08_results/fresh_distribution_comparison_samples32.json")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    main(parser.parse_args())
