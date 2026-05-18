# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch
import torch.nn.functional as F



def _flatten_samples(samples: torch.Tensor) -> torch.Tensor:
    return samples.flatten(2)


def _off_diagonal_mean(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    assert matrix.dim() == 3 and matrix.shape[1] == matrix.shape[2]
    n = matrix.shape[1]
    if n <= 1:
        return matrix.new_tensor(0.0)
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device).unsqueeze(0)
    return matrix.masked_select(mask).mean().clamp_min(eps)


def sample_diversity_l1(samples: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if samples.shape[1] <= 1:
        return samples.new_tensor(0.0)
    flat = _flatten_samples(samples)
    diff = (flat[:, :, None, :] - flat[:, None, :, :]).abs().mean(dim=-1)
    return _off_diagonal_mean(diff, eps=eps)


def diversity_matching_loss(
    student_samples: torch.Tensor,
    teacher_samples: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    student_div = sample_diversity_l1(student_samples, eps=eps)
    teacher_div = sample_diversity_l1(teacher_samples.detach(), eps=eps)
    deficit = F.relu(teacher_div - student_div)
    loss = (deficit / teacher_div.detach().clamp_min(eps)).pow(2)
    stats = {
        'diversity_loss': float(loss.detach().item()),
        'student_diversity': float(student_div.detach().item()),
        'teacher_diversity': float(teacher_div.detach().item()),
        'diversity_ratio': float((student_div / teacher_div.detach().clamp_min(eps)).detach().item()),
    }
    return loss, stats


def energy_distance_to_teacher_loss(
    student_samples: torch.Tensor,
    teacher_samples: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    assert student_samples.dim() == teacher_samples.dim(), 'student and teacher samples need matching ranks'
    assert student_samples.shape[0] == teacher_samples.shape[0], 'batch size mismatch'
    assert student_samples.shape[2:] == teacher_samples.shape[2:], 'sample trailing shape mismatch'

    y = _flatten_samples(student_samples)
    t = _flatten_samples(teacher_samples.detach())

    cross = torch.cdist(y, t, p=2).mean()
    yy = torch.cdist(y, y, p=2).mean()
    tt = torch.cdist(t, t, p=2).mean()
    loss = 2.0 * cross - yy - tt

    _, div_stats = diversity_matching_loss(student_samples, teacher_samples, eps=eps)
    stats = {
        'loss': float(loss.detach().item()),
        'energy_cross': float(cross.detach().item()),
        'energy_student_student': float(yy.detach().item()),
        'energy_teacher_teacher': float(tt.detach().item()),
        **div_stats,
    }
    return loss, stats


def combined_drift_diversity_loss(
    student_samples: torch.Tensor,
    teacher_samples: torch.Tensor,
    eta: float = 0.1,
    sigma: Optional[float] = None,
    lambda_diversity: float = 0.1,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    drift_loss, drift_stats = rbf_drift_to_teacher_loss(
        student_samples,
        teacher_samples,
        eta=eta,
        sigma=sigma,
        eps=eps,
    )
    div_loss, div_stats = diversity_matching_loss(student_samples, teacher_samples, eps=eps)
    loss = drift_loss + lambda_diversity * div_loss
    stats = {
        **drift_stats,
        **div_stats,
        'loss': float(loss.detach().item()),
        'drift_loss': float(drift_loss.detach().item()),
        'lambda_diversity': float(lambda_diversity),
    }
    return loss, stats



def _pairwise_l1_matrix(samples: torch.Tensor) -> torch.Tensor:
    flat = _flatten_samples(samples)
    return (flat[:, :, None, :] - flat[:, None, :, :]).abs().mean(dim=-1)


def _off_diagonal_values(matrix: torch.Tensor) -> torch.Tensor:
    assert matrix.dim() == 3 and matrix.shape[1] == matrix.shape[2]
    n = matrix.shape[1]
    if n <= 1:
        return matrix.new_zeros(1)
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device).unsqueeze(0)
    return matrix.masked_select(mask)


def noise_conditioning_loss(
    student_samples: torch.Tensor,
    z: torch.Tensor,
    teacher_samples: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    # Encourage different latent noises to produce proportionally different outputs.
    # The teacher particle diversity sets the output distance scale.
    y_dist = _pairwise_l1_matrix(student_samples)
    z_dist = _pairwise_l1_matrix(z)
    teacher_div = sample_diversity_l1(teacher_samples.detach(), eps=eps)

    y_vals = _off_diagonal_values(y_dist)
    z_vals = _off_diagonal_values(z_dist).detach()
    z_target = z_vals / z_vals.mean().clamp_min(eps) * teacher_div.detach()
    loss = F.smooth_l1_loss(y_vals, z_target)

    y_centered = y_vals.detach() - y_vals.detach().mean()
    z_centered = z_vals - z_vals.mean()
    corr = (y_centered * z_centered).mean() / (
        y_centered.pow(2).mean().sqrt() * z_centered.pow(2).mean().sqrt() + eps
    )
    stats = {
        'noise_loss': float(loss.detach().item()),
        'noise_output_corr': float(corr.detach().item()),
        'noise_target_diversity': float(teacher_div.detach().item()),
    }
    return loss, stats


def combined_drift_diversity_noise_loss(
    student_samples: torch.Tensor,
    teacher_samples: torch.Tensor,
    z: torch.Tensor,
    eta: float = 0.1,
    sigma: Optional[float] = None,
    lambda_diversity: float = 0.1,
    lambda_noise: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    base_loss, base_stats = combined_drift_diversity_loss(
        student_samples,
        teacher_samples,
        eta=eta,
        sigma=sigma,
        lambda_diversity=lambda_diversity,
        eps=eps,
    )
    noise_loss, noise_stats = noise_conditioning_loss(student_samples, z, teacher_samples, eps=eps)
    loss = base_loss + lambda_noise * noise_loss
    stats = {
        **base_stats,
        **noise_stats,
        'loss': float(loss.detach().item()),
        'base_loss': float(base_loss.detach().item()),
        'lambda_noise': float(lambda_noise),
    }
    return loss, stats

def auto_sigma(student_samples: torch.Tensor, teacher_samples: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Median pairwise distance between student and teacher particles.
    Inputs:
      student_samples: (B, S, ...)
      teacher_samples: (B, K, ...)
    Returns scalar sigma tensor detached from graph.
    """
    with torch.no_grad():
        s = student_samples.detach().flatten(2)
        t = teacher_samples.detach().flatten(2)
        dist = torch.cdist(s, t, p=2)
        sigma = torch.median(dist)
        sigma = torch.clamp(sigma, min=eps)
    return sigma


def rbf_drift_to_teacher_loss(
    student_samples: torch.Tensor,
    teacher_samples: torch.Tensor,
    eta: float = 0.1,
    sigma: Optional[float] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """
    Pure Drift-to-teacher particle loss.

    For each condition independently:
      k(y_i, t_j) = exp(-||y_i - t_j||^2 / (2 sigma^2))
      attraction_i = sum_j k_ij * (t_j - y_i) / (sum_j k_ij + eps)
      target_i = stop_gradient(y_i + eta * attraction_i)
      loss = mean_i ||y_i - target_i||^2

    No GT loss, no teacher L1/L2 matching, no online teacher calls.
    """
    assert student_samples.dim() == teacher_samples.dim(), "student and teacher samples need matching ranks"
    assert student_samples.shape[0] == teacher_samples.shape[0], "batch size mismatch"
    assert student_samples.shape[2:] == teacher_samples.shape[2:], "sample trailing shape mismatch"

    y = student_samples
    t = teacher_samples.detach()
    B, S = y.shape[:2]
    K = t.shape[1]
    y_flat = y.detach().flatten(2)
    t_flat = t.flatten(2)

    dist2 = torch.cdist(y_flat, t_flat, p=2).pow(2)
    if sigma is None or sigma <= 0:
        sigma_t = auto_sigma(y, t, eps=eps).to(y.device)
    else:
        sigma_t = torch.tensor(float(sigma), device=y.device, dtype=y.dtype)
    sigma2 = torch.clamp(sigma_t.pow(2), min=eps)

    weights = torch.exp(-dist2 / (2.0 * sigma2))
    weights = weights / (weights.sum(dim=2, keepdim=True) + eps)

    teacher_flat = t_flat[:, None, :, :]
    student_flat = y_flat[:, :, None, :]
    drift_flat = (weights[..., None] * (teacher_flat - student_flat)).sum(dim=2)
    drift = drift_flat.reshape_as(y)
    target = (y + eta * drift).detach()
    loss = F.mse_loss(y, target)

    stats = {
        "loss": float(loss.detach().item()),
        "sigma": float(sigma_t.detach().item()),
        "drift_norm": float(drift.detach().flatten(2).norm(dim=2).mean().item()),
        "student_mean": float(y.detach().mean().item()),
        "teacher_mean": float(t.mean().item()),
        "student_diversity": float(sample_diversity_l1(y.detach(), eps=eps).item()),
        "teacher_diversity": float(sample_diversity_l1(t, eps=eps).item()),
        "diversity_ratio": float(
            (sample_diversity_l1(y.detach(), eps=eps) / sample_diversity_l1(t, eps=eps).clamp_min(eps)).item()
        ),
        "batch_size": B,
        "num_student_samples": S,
        "num_teacher_samples": K,
    }
    return loss, stats
