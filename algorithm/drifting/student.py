# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from algorithm.diffstg.ugnet import UGnet


class DriftTeacherStudent(nn.Module):
    """
    One-step conditional generator.

    Inputs use the DiffSTG cache convention:
      z_future: (B, F, V, T_p)
      x_masked: (B, F, V, T_h + T_p), normalized, future portion zero
      pos_w/pos_d: kept for compatibility with UGnet condition tuple

    Output:
      future sample: (B, T_p, V, F), normalized

    NFE is exactly 1: one UGnet forward call per generated sample.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = UGnet(config)

    def forward_single(self, z_future, x_masked, pos_w, pos_d):
        B, F, V, T_total = x_masked.shape
        T_p = self.config.T_p
        T_h = self.config.T_h
        assert z_future.shape == (B, F, V, T_p), (
            f"z_future shape {tuple(z_future.shape)} does not match "
            f"expected {(B, F, V, T_p)}"
        )

        z_full = torch.zeros(B, F, V, T_total, device=z_future.device, dtype=z_future.dtype)
        z_full[:, :, :, T_h:] = z_future
        t = torch.zeros(B, device=z_future.device, dtype=torch.long)
        out_full = self.backbone(z_full, t, (x_masked, pos_w, pos_d))
        out_future = out_full[:, :, :, -T_p:]
        return out_future.transpose(1, 3).contiguous()

    def forward(self, x_masked, pos_w, pos_d, num_samples=1, z=None):
        B, F, V, _ = x_masked.shape
        T_p = self.config.T_p
        if z is None:
            z = torch.randn(B, num_samples, F, V, T_p, device=x_masked.device)
        else:
            assert z.shape == (B, num_samples, F, V, T_p), (
                f"z shape {tuple(z.shape)} does not match "
                f"expected {(B, num_samples, F, V, T_p)}"
            )

        x_rep = x_masked.unsqueeze(1).repeat(1, num_samples, 1, 1, 1).reshape(B * num_samples, F, V, -1)
        pos_w_rep = pos_w.unsqueeze(1).repeat(1, num_samples, *([1] * (pos_w.dim() - 1))).reshape(B * num_samples, *pos_w.shape[1:])
        pos_d_rep = pos_d.unsqueeze(1).repeat(1, num_samples, *([1] * (pos_d.dim() - 1))).reshape(B * num_samples, *pos_d.shape[1:])
        z_rep = z.reshape(B * num_samples, F, V, T_p)

        y = self.forward_single(z_rep, x_rep, pos_w_rep, pos_d_rep)
        return y.reshape(B, num_samples, T_p, V, F)
