# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from timeit import default_timer as timer

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.checkpoint import load_teacher_checkpoint
from algorithm.diffstg.model import DiffSTG
from algorithm.drifting.student import DriftTeacherStudent
from eval_drift_teacher_student import build_config_from_checkpoint as build_student_config
from eval_teacher_pems08 import apply_checkpoint_config, masked_mae_np, masked_rmse_np, sample_crps_np
from train import default_config, setup_seed
from utils.common_utils import to_device


def pairwise_l1(a, b):
    af = a.reshape(a.shape[0], -1).astype(np.float64)
    bf = b.reshape(b.shape[0], -1).astype(np.float64)
    return np.abs(af[:, None, :] - bf[None, :, :]).mean(axis=2)


def offdiag_mean(cost):
    if cost.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(cost.shape[0], dtype=bool)
    return float(cost[mask].mean())


def distribution_metrics(model_samples, teacher_samples):
    cross = pairwise_l1(model_samples, teacher_samples)
    mm = pairwise_l1(model_samples, model_samples)
    tt = pairwise_l1(teacher_samples, teacher_samples)
    model_div = offdiag_mean(mm)
    teacher_div = offdiag_mean(tt)
    energy = float(2.0 * cross.mean() - model_div - teacher_div)
    row_ind, col_ind = linear_sum_assignment(cross)
    emd = float(cross[row_ind, col_ind].mean())
    return energy, emd, model_div, teacher_div


def load_teacher_model(path, device, batch_size):
    ckpt = load_teacher_checkpoint(path, map_location='cpu')
    config = default_config('PEMS08')
    config = apply_checkpoint_config(config, ckpt)
    config.model.sample_steps = 100
    config.n_samples = 32
    config.batch_size = batch_size
    config.is_test = False
    config.device = device
    config.model.device = device
    clean_data = CleanDataset(config)
    scaler = ckpt.get('scaler', {})
    if 'mean' in scaler and 'std' in scaler:
        clean_data.mean = float(scaler['mean'])
        clean_data.std = float(scaler['std'])
    config.model.A = clean_data.adj
    model = DiffSTG(config.model).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.set_sample_strategy('ddim_multi')
    model.eval()
    return model, clean_data, config


def load_student_model(path, device):
    ckpt = torch.load(path, map_location='cpu')
    config = build_student_config(ckpt, device)
    clean_data = CleanDataset(config)
    norm = ckpt.get('normalization', {})
    if 'mean' in norm and 'std' in norm:
        clean_data.mean = float(norm['mean'])
        clean_data.std = float(norm['std'])
    config.model.A = clean_data.adj
    model = DriftTeacherStudent(config.model).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model


def make_acc(spec):
    acc = dict(spec)
    acc.update({
        'crps_sum': 0.0,
        'mae_sum': 0.0,
        'rmse_sq_sum': 0.0,
        'n_examples': 0,
        'energy_values': [],
        'emd_values': [],
        'model_div_values': [],
        'teacher_div_values': [],
        'inference_time_sec': 0.0,
        'num_eval_batches': 0,
    })
    return acc


def update_gt(acc, samples, target):
    bsz = target.shape[0]
    y_mean = samples.mean(axis=1)
    acc['crps_sum'] += sample_crps_np(samples, target) * bsz
    acc['mae_sum'] += masked_mae_np(target, y_mean, 0.0) * bsz
    acc['rmse_sq_sum'] += (masked_rmse_np(target, y_mean, 0.0) ** 2) * bsz
    acc['n_examples'] += bsz


def update_dist(acc, samples, teacher_ref, self_ref=False):
    for i in range(samples.shape[0]):
        if self_ref:
            cost = pairwise_l1(teacher_ref[i], teacher_ref[i])
            div = offdiag_mean(cost)
            energy, emd, model_div, teacher_div = 0.0, 0.0, div, div
        else:
            energy, emd, model_div, teacher_div = distribution_metrics(samples[i], teacher_ref[i])
        acc['energy_values'].append(energy)
        acc['emd_values'].append(emd)
        acc['model_div_values'].append(model_div)
        acc['teacher_div_values'].append(teacher_div)


def teacher_predict(model, config, clean_data, x_masked, pos_w, pos_d, steps):
    model.set_ddim_sample_steps(steps)
    x_hat = model((x_masked, pos_w, pos_d), 32)
    y = x_hat[:, :, :, :, -config.model.T_p:].transpose(2, 4).contiguous()
    return clean_data.reverse_normalization(y).detach().cpu().numpy().clip(0.0, None)


def student_predict(model, clean_data, x_masked, pos_w, pos_d, num_samples):
    y = model(x_masked, pos_w, pos_d, num_samples)
    return clean_data.reverse_normalization(y).detach().cpu().numpy().clip(0.0, None)


def finalize(rows):
    base_time = next(r['inference_time_sec'] for r in rows if r['method'] == 'DiffSTG Teacher (100-step)')
    final = []
    for r in rows:
        n = max(r['n_examples'], 1)
        model_div = float(np.mean(r['model_div_values'])) if r['model_div_values'] else 0.0
        teacher_div = float(np.mean(r['teacher_div_values'])) if r['teacher_div_values'] else 0.0
        final.append({
            'method': r['method'],
            'NFE': r['NFE'],
            'num_samples': r['num_samples'],
            'teacher_reference_samples': r['teacher_reference_samples'],
            'CRPS': float(r['crps_sum'] / n),
            'MAE': float(r['mae_sum'] / n),
            'RMSE': float((r['rmse_sq_sum'] / n) ** 0.5),
            'sample_diversity': model_div,
            'teacher100_diversity': teacher_div,
            'relative_diversity_vs_teacher100': float(model_div / teacher_div) if teacher_div > 0 else None,
            'EnergyDistance_to_teacher100': float(np.mean(r['energy_values'])) if r['energy_values'] else None,
            'EMD_to_teacher100': float(np.mean(r['emd_values'])) if r['emd_values'] else None,
            'inference_time_sec': float(r['inference_time_sec']),
            'speedup_vs_teacher100': float(base_time / r['inference_time_sec']) if r['inference_time_sec'] else None,
            'num_eval_examples': int(r['n_examples']),
            'num_eval_batches': int(r['num_eval_batches']),
            'checkpoint': r['checkpoint'],
        })
    return final



def write_result(rows, output_dir, suffix=""):
    os.makedirs(output_dir, exist_ok=True)
    result = finalize(rows)
    stem = "fresh_stream_emd_comparison" + suffix
    json_path = os.path.join(output_dir, stem + ".json")
    csv_path = os.path.join(output_dir, stem + ".csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0].keys()))
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(result, indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")


def main(args):
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, clean_data, config = load_teacher_model(args.teacher_checkpoint, device, args.batch_size)
    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False)

    student_specs = [
        ("Drift Student cache8", args.student_cache8_checkpoint, 8),
        ("Drift Student cache8 Energy FT", args.student_cache8_energy_checkpoint, 8),
        ("Drift Student cache32", args.student_cache32_checkpoint, 32),
        ("Drift Student cache32 Energy FT (lr 5e-5)", args.student_cache32_energy_lr5_checkpoint, 32),
        ("Drift Student cache32 Energy FT (lr 1e-5)", args.student_cache32_energy_lr1_checkpoint, 32),
    ]
    students = [(name, load_student_model(path, device), path, ns) for name, path, ns in student_specs]

    rows = [make_acc({"method": "DiffSTG Teacher (100-step)", "NFE": 100, "num_samples": 32, "teacher_reference_samples": 32, "checkpoint": args.teacher_checkpoint})]
    for name, _, path, ns in students:
        rows.append(make_acc({"method": name, "NFE": 1, "num_samples": ns, "teacher_reference_samples": ns, "checkpoint": path}))

    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            if args.max_eval_batches is not None and batch_i >= args.max_eval_batches:
                break
            future, history, pos_w, pos_d = to_device(batch, device)
            x_masked_bt = torch.cat((history, torch.zeros_like(future)), dim=1).to(device)
            x_masked = x_masked_bt.transpose(1, 3).contiguous()
            target = clean_data.reverse_normalization(future).detach().cpu().numpy()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = timer()
            teacher100 = teacher_predict(teacher, config, clean_data, x_masked, pos_w, pos_d, 100)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            rows[0]["inference_time_sec"] += timer() - start
            rows[0]["num_eval_batches"] += 1
            update_gt(rows[0], teacher100, target)
            update_dist(rows[0], teacher100, teacher100, self_ref=True)

            for row, (_, model, _, ns) in zip(rows[1:], students):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = timer()
                pred = student_predict(model, clean_data, x_masked, pos_w, pos_d, ns)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row["inference_time_sec"] += timer() - start
                row["num_eval_batches"] += 1
                update_gt(row, pred, target)
                update_dist(row, pred, teacher100[:, :ns, ...])

            if batch_i == 0 or (batch_i + 1) % args.log_interval == 0:
                print(f"phase=teacher100_students batch={batch_i + 1}")

    write_result(rows, args.output_dir, suffix="_no_teacher40")
    if args.skip_teacher40:
        write_result(rows, args.output_dir)
        return

    teacher40_row = make_acc({"method": "DiffSTG Teacher (40-step)", "NFE": 40, "num_samples": 32, "teacher_reference_samples": 32, "checkpoint": args.teacher_checkpoint})
    loader40 = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch_i, batch in enumerate(loader40):
            if args.max_eval_batches is not None and batch_i >= args.max_eval_batches:
                break
            future, history, pos_w, pos_d = to_device(batch, device)
            x_masked_bt = torch.cat((history, torch.zeros_like(future)), dim=1).to(device)
            x_masked = x_masked_bt.transpose(1, 3).contiguous()
            target = clean_data.reverse_normalization(future).detach().cpu().numpy()

            teacher100 = teacher_predict(teacher, config, clean_data, x_masked, pos_w, pos_d, 100)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = timer()
            teacher40 = teacher_predict(teacher, config, clean_data, x_masked, pos_w, pos_d, 40)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            teacher40_row["inference_time_sec"] += timer() - start
            teacher40_row["num_eval_batches"] += 1
            update_gt(teacher40_row, teacher40, target)
            update_dist(teacher40_row, teacher40, teacher100)

            if batch_i == 0 or (batch_i + 1) % args.log_interval == 0:
                print(f"phase=teacher40_last batch={batch_i + 1}")

    write_result([rows[0], teacher40_row] + rows[1:], args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fresh PEMS08 stream eval with Energy Distance and EMD.')
    parser.add_argument('--teacher_checkpoint', default='outputs/pems08_teacher/checkpoints/best.pt')
    parser.add_argument('--student_cache8_checkpoint', default='outputs/pems08_student_drift_teacher_s8_antisymmetric/checkpoints/best.pt')
    parser.add_argument('--student_cache8_energy_checkpoint', default='outputs/pems08_student_energy_finetune_full/checkpoints/best.pt')
    parser.add_argument('--student_cache32_checkpoint', default='outputs/pems08_student_drift_teacher_s32_antisymmetric/checkpoints/best.pt')
    parser.add_argument('--student_cache32_energy_lr5_checkpoint', default='outputs/pems08_student_energy_finetune_s32_full/checkpoints/best.pt')
    parser.add_argument('--student_cache32_energy_lr1_checkpoint', default='outputs/pems08_student_energy_finetune_s32_lr1e5_ep5/checkpoints/best.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', default='outputs/pems08_fresh_stream_emd_eval')
    parser.add_argument('--max_eval_batches', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--num_threads', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--skip_teacher40', action='store_true')
    main(parser.parse_args())
