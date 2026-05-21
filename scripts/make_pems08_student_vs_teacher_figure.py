#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from algorithm.dataset import TrafficDataset  # noqa: E402
from generate_fresh_pems08_samples import load_student, load_teacher  # noqa: E402
from train import setup_seed  # noqa: E402

DEFAULT_STUDENT_CKPT = PROJECT_DIR / "outputs/pems08_student_drift_teacher_s32_antisymmetric_diversity_finetune/checkpoints/best.pt"
USER_TEACHER_CKPT = PROJECT_DIR / "outputs/pems08_student_drift_teacher/checkpoints/best.pt"
TRUE_TEACHER_CKPT = PROJECT_DIR / "outputs/pems08_teacher/checkpoints/best.pt"




def _sample_tensor_from_shard(shard):
    if "samples" in shard:
        return shard["samples"]
    if "teacher_samples" in shard:
        return shard["teacher_samples"]
    return shard["student_samples"]


def load_saved_example(student_sample_dir, teacher_sample_dir, example_offset=None, candidate_count=256, node_idx=None):
    student_sample_dir = Path(student_sample_dir)
    teacher_sample_dir = Path(teacher_sample_dir)
    with open(student_sample_dir / "index.json", "r", encoding="utf-8") as f:
        student_index = json.load(f)
    with open(teacher_sample_dir / "index.json", "r", encoding="utf-8") as f:
        teacher_index = json.load(f)
    if len(student_index["shards"]) != len(teacher_index["shards"]):
        raise ValueError("student/teacher sample dirs have different shard counts")

    if example_offset is None:
        scores = []
        checked = 0
        for sp, tp in zip(student_index["shards"], teacher_index["shards"]):
            ss = torch.load(sp, map_location="cpu")
            ts = torch.load(tp, map_location="cpu")
            if not torch.equal(ss["indices"], ts["indices"]):
                raise ValueError(f"index mismatch between {sp} and {tp}")
            gt = ts["gt_future"]
            hist = ts.get("history")
            for b in range(gt.shape[0]):
                if checked >= candidate_count:
                    break
                full = gt[b] if hist is None else torch.cat([hist[b], gt[b]], dim=0)
                scores.append((float(gt[b].mean().item()), float(full.max().item() - full.min().item()), int(ts["indices"][b].item())))
                checked += 1
            if checked >= candidate_count:
                break
        means = np.asarray([x[0] for x in scores])
        ranges = np.asarray([x[1] for x in scores])
        offsets = np.asarray([x[2] for x in scores])
        median_mean = float(np.median(means))
        candidates = np.where(ranges >= float(np.percentile(ranges, 50)))[0]
        best_i = candidates[np.argmin(np.abs(means[candidates] - median_mean))]
        example_offset = int(offsets[best_i])
        example_reason = f"saved_auto_median_future_mean_among_{len(scores)}_candidates"
    else:
        example_offset = int(example_offset)
        example_reason = "manual_saved_index"

    for sp, tp in zip(student_index["shards"], teacher_index["shards"]):
        ss = torch.load(sp, map_location="cpu")
        ts = torch.load(tp, map_location="cpu")
        if not torch.equal(ss["indices"], ts["indices"]):
            raise ValueError(f"index mismatch between {sp} and {tp}")
        matches = (ts["indices"] == example_offset).nonzero(as_tuple=False)
        if matches.numel() == 0:
            continue
        b = int(matches[0].item())
        student_samples = _sample_tensor_from_shard(ss)[b].numpy()
        teacher_samples = _sample_tensor_from_shard(ts)[b].numpy()
        future_orig = ts["gt_future"][b]
        history_orig = ts.get("history")
        if history_orig is not None:
            history_orig = history_orig[b]
            gt_full = torch.cat([history_orig, future_orig], dim=0)
        else:
            history_orig = None
            gt_full = future_orig
        if node_idx is None:
            full_for_range = gt_full[:, :, 0]
            ranges = (full_for_range.max(dim=0).values - full_for_range.min(dim=0).values).numpy()
            node_idx = int(np.argmax(ranges))
            node_reason = "saved_auto_largest_gt_range"
        else:
            node_idx = int(node_idx)
            node_reason = "manual"
        data_indices = ts.get("data_indices")
        data_index = int(data_indices[b].item()) if data_indices is not None else example_offset
        print(f"loaded_from_saved_samples=True")
        print(f"student_sample_dir={student_sample_dir}")
        print(f"teacher_sample_dir={teacher_sample_dir}")
        return {
            "student_samples": student_samples,
            "teacher_samples": teacher_samples,
            "gt_series": gt_full[:, node_idx, 0].numpy(),
            "history_len": int(history_orig.shape[0]) if history_orig is not None else 0,
            "future_len": int(future_orig.shape[0]),
            "example_offset": example_offset,
            "example_reason": example_reason,
            "data_index": data_index,
            "node_idx": node_idx,
            "node_reason": node_reason,
            "student_label_source": str(student_sample_dir),
            "teacher_label_source": str(teacher_sample_dir),
        }
    raise ValueError(f"example_offset={example_offset} not found in saved sample dirs")

def tensor_to_original(x, clean_data):
    return x * float(clean_data.std) + float(clean_data.mean)


def build_test_dataset(clean_data, config):
    return TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)


def choose_example(dataset, clean_data, candidate_count, explicit_offset=None):
    if explicit_offset is not None:
        return int(explicit_offset), "manual"
    n = min(int(candidate_count), len(dataset))
    means = []
    ranges = []
    for idx in range(n):
        future, history, _, _ = dataset[idx]
        future = torch.as_tensor(future).float()
        history = torch.as_tensor(history).float()
        full = tensor_to_original(torch.cat([history, future], dim=0), clean_data)
        future_orig = tensor_to_original(future, clean_data)
        means.append(float(future_orig.mean().item()))
        ranges.append(float(full.max().item() - full.min().item()))
    means_np = np.asarray(means)
    ranges_np = np.asarray(ranges)
    median_mean = float(np.median(means_np))
    # Prefer a median-flow example, but avoid visually flat cases.
    range_threshold = float(np.percentile(ranges_np, 50))
    candidates = np.where(ranges_np >= range_threshold)[0]
    if len(candidates) == 0:
        candidates = np.arange(n)
    best = candidates[np.argmin(np.abs(means_np[candidates] - median_mean))]
    return int(best), f"auto_median_future_mean_among_{n}_candidates"


def choose_node(history_orig, future_orig, explicit_node=None):
    if explicit_node is not None:
        return int(explicit_node), "manual"
    full = torch.cat([history_orig, future_orig], dim=0)[:, :, 0]
    ranges = (full.max(dim=0).values - full.min(dim=0).values).cpu().numpy()
    return int(np.argmax(ranges)), "auto_largest_24step_range"


def prepare_single_example(dataset, clean_data, config, example_offset, device):
    future, history, pos_w, pos_d = dataset[example_offset]
    future = torch.as_tensor(future).float()
    history = torch.as_tensor(history).float()
    pos_w = torch.as_tensor(pos_w).long()
    pos_d = torch.as_tensor(pos_d).long()
    future_b = future.unsqueeze(0).to(device)
    history_b = history.unsqueeze(0).to(device)
    pos_w_b = pos_w.unsqueeze(0).to(device)
    pos_d_b = pos_d.unsqueeze(0).to(device)
    x_masked = torch.cat([history_b, torch.zeros_like(future_b)], dim=1).transpose(1, 3).contiguous()
    history_orig = tensor_to_original(history, clean_data).cpu()
    future_orig = tensor_to_original(future, clean_data).cpu()
    data_index = int(config.data.test_start_idx + config.model.T_p + example_offset)
    return future_b, history_b, pos_w_b, pos_d_b, x_masked, history_orig, future_orig, data_index


def sample_student(model, clean_data, x_masked, pos_w, pos_d, num_samples, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = timer()
    with torch.no_grad():
        y = model(x_masked, pos_w, pos_d, num_samples)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = timer() - start
    y = clean_data.reverse_normalization(y).detach().cpu().clamp_min(0.0)
    return y[0].numpy(), elapsed


def sample_teacher(model, clean_data, config, x_masked, pos_w, pos_d, num_samples, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = timer()
    with torch.no_grad():
        x_hat = model((x_masked, pos_w, pos_d), num_samples)
        y = x_hat[:, :, :, :, -config.model.T_p:].transpose(2, 4).contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = timer() - start
    y = clean_data.reverse_normalization(y).detach().cpu().clamp_min(0.0)
    return y[0].numpy(), elapsed


def band_stats(samples, node_idx, feature_idx=0, q_low=10, q_high=90):
    series = samples[:, :, node_idx, feature_idx]
    return {
        "mean": series.mean(axis=0),
        "low": np.percentile(series, q_low, axis=0),
        "high": np.percentile(series, q_high, axis=0),
    }


def plot_panel(ax, title, label, gt_x, gt_y, future_x, stats, show_ylabel=False):
    band = ax.fill_between(
        future_x,
        stats["low"],
        stats["high"],
        color="#4CAF50",
        alpha=0.28,
        linewidth=0,
        label=label,
    )
    mean_line, = ax.plot(future_x, stats["mean"], color="#0B5D1E", linewidth=2.4, label="Predictive mean")
    gt = ax.scatter(gt_x, gt_y, s=34, color="#1565C0", edgecolor="white", linewidth=0.6, zorder=4, label="Ground Truth")
    ax.axvline(11.5, color="black", linestyle="--", linewidth=1.4)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.text(0.98, 0.96, "PEMS08", transform=ax.transAxes, ha="right", va="top", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time step", fontsize=15)
    if show_ylabel:
        ax.set_ylabel("Traffic Flow", fontsize=15)
    ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(handles=[band, gt], loc="upper left", frameon=True, framealpha=0.92, fontsize=12)


def main(args):
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    if args.student_sample_dir and args.teacher_sample_dir:
        saved = load_saved_example(
            args.student_sample_dir,
            args.teacher_sample_dir,
            example_offset=args.example_offset,
            candidate_count=args.candidate_count,
            node_idx=args.node_idx,
        )
        student_samples = saved["student_samples"]
        teacher_samples = saved["teacher_samples"]
        gt_series = saved["gt_series"]
        history_len = saved["history_len"]
        future_len = saved["future_len"]
        gt_x = np.arange(history_len + future_len)
        future_x = np.arange(history_len, history_len + future_len)
        example_offset = saved["example_offset"]
        example_reason = saved["example_reason"]
        data_index = saved["data_index"]
        node_idx = saved["node_idx"]
        node_reason = saved["node_reason"]
        student_time = 0.0
        teacher_time = 0.0
        teacher_checkpoint = "saved_samples_no_inference"
    elif args.student_sample_dir or args.teacher_sample_dir:
        raise ValueError("Provide both --student_sample_dir and --teacher_sample_dir, or neither.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        student_args = SimpleNamespace(checkpoint=str(args.student_checkpoint))
        student_model, student_clean, student_config, _ = load_student(student_args, device)

        teacher_checkpoint = Path(args.teacher_checkpoint)
        teacher_args = SimpleNamespace(
            checkpoint=str(teacher_checkpoint),
            sample_steps=args.teacher_sample_steps,
            num_samples=args.num_samples,
            batch_size=1,
        )
        try:
            teacher_model, teacher_clean, teacher_config, _ = load_teacher(teacher_args, device)
        except RuntimeError as exc:
            fallback = Path(args.teacher_fallback_checkpoint)
            if args.allow_teacher_fallback and fallback.exists():
                print("WARNING: requested teacher checkpoint could not be loaded as DiffSTG teacher.")
                print(str(exc))
                print(f"Using fallback true teacher checkpoint: {fallback}")
                teacher_checkpoint = fallback
                teacher_args.checkpoint = str(fallback)
                teacher_model, teacher_clean, teacher_config, _ = load_teacher(teacher_args, device)
            else:
                raise

        if abs(float(student_clean.mean) - float(teacher_clean.mean)) > 1e-5 or abs(float(student_clean.std) - float(teacher_clean.std)) > 1e-5:
            print("WARNING: student/teacher normalization differs; using each model's own inverse normalization for predictions.")

        dataset = build_test_dataset(teacher_clean, teacher_config)
        example_offset, example_reason = choose_example(dataset, teacher_clean, args.candidate_count, args.example_offset)
        future_b, history_b, pos_w, pos_d, x_masked, history_orig, future_orig, data_index = prepare_single_example(
            dataset, teacher_clean, teacher_config, example_offset, device
        )
        node_idx, node_reason = choose_node(history_orig, future_orig, args.node_idx)

        student_samples, student_time = sample_student(student_model, student_clean, x_masked, pos_w, pos_d, args.num_samples, device)
        teacher_samples, teacher_time = sample_teacher(
            teacher_model, teacher_clean, teacher_config, x_masked, pos_w, pos_d, args.num_samples, device
        )

        gt_series = torch.cat([history_orig, future_orig], dim=0)[:, node_idx, 0].numpy()
        gt_x = np.arange(teacher_config.model.T_h + teacher_config.model.T_p)
        future_x = np.arange(teacher_config.model.T_h, teacher_config.model.T_h + teacher_config.model.T_p)

    student_stats = band_stats(student_samples, node_idx, q_low=args.q_low, q_high=args.q_high)
    teacher_stats = band_stats(teacher_samples, node_idx, q_low=args.q_low, q_high=args.q_high)

    y_values = [gt_series, student_stats["low"], student_stats["high"], student_stats["mean"], teacher_stats["low"], teacher_stats["high"], teacher_stats["mean"]]
    ymin = min(float(np.min(v)) for v in y_values)
    ymax = max(float(np.max(v)) for v in y_values)
    margin = max((ymax - ymin) * 0.12, 5.0)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharex=True, sharey=True)
    plot_panel(axes[0], "(a) Ours (1-step student)", "Ours (1-step student)", gt_x, gt_series, future_x, student_stats, show_ylabel=True)
    plot_panel(axes[1], "(b) Teacher (DiffSTG)", "Teacher (DiffSTG)", gt_x, gt_series, future_x, teacher_stats, show_ylabel=False)
    for ax in axes:
        ax.set_xlim(-0.5, len(gt_x) - 0.5)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_xticks(np.arange(0, len(gt_x), 3))
    fig.tight_layout(w_pad=2.4)

    output_png = Path(args.output_png)
    output_pdf = Path(args.output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"selected_example_offset={example_offset} reason={example_reason}")
    print(f"selected_data_index={data_index}")
    print(f"selected_node_idx={node_idx} reason={node_reason}")
    print(f"student_checkpoint={args.student_checkpoint}")
    print(f"teacher_checkpoint_used={teacher_checkpoint}")
    print(f"student_samples_shape={student_samples.shape} inference_time_sec={student_time:.4f}")
    print(f"teacher_samples_shape={teacher_samples.shape} inference_time_sec={teacher_time:.4f}")
    print(f"band_percentiles={args.q_low}-{args.q_high}")
    print(f"saved_png={output_png}")
    print(f"saved_pdf={output_pdf}")


def parse_args():
    parser = argparse.ArgumentParser(description="Make a two-panel PEMS08 student-vs-teacher probabilistic band figure.")
    parser.add_argument("--student_sample_dir", type=Path, default=None, help="Use saved metric samples instead of running student inference.")
    parser.add_argument("--teacher_sample_dir", type=Path, default=None, help="Use saved metric samples instead of running teacher inference.")
    parser.add_argument("--student_checkpoint", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--teacher_checkpoint", type=Path, default=USER_TEACHER_CKPT)
    parser.add_argument("--teacher_fallback_checkpoint", type=Path, default=TRUE_TEACHER_CKPT)
    parser.add_argument("--allow_teacher_fallback", action="store_true", default=True)
    parser.add_argument("--teacher_sample_steps", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--example_offset", type=int, default=None)
    parser.add_argument("--candidate_count", type=int, default=256)
    parser.add_argument("--node_idx", type=int, default=None)
    parser.add_argument("--q_low", type=float, default=10.0)
    parser.add_argument("--q_high", type=float, default=90.0)
    parser.add_argument("--output_png", type=Path, default=PROJECT_DIR / "figures/pems08_student_vs_teacher.png")
    parser.add_argument("--output_pdf", type=Path, default=PROJECT_DIR / "figures/pems08_student_vs_teacher.pdf")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num_threads", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
