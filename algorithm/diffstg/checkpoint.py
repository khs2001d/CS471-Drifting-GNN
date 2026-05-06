# -*- coding: utf-8 -*-
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch


def _plain_value(value: Any) -> Any:
    """Convert config values into checkpoint-friendly Python objects."""
    if isinstance(value, dict):
        return {k: _plain_value(v) for k, v in value.items() if k != "logger"}
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def config_to_plain(config: Any) -> Dict[str, Any]:
    return _plain_value(dict(config))


def save_teacher_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_metric: Dict[str, Any],
    args: Dict[str, Any],
    config: Any,
    scaler: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_validation_metric": _plain_value(deepcopy(best_metric)),
        "args": _plain_value(deepcopy(args)),
        "config": config_to_plain(config),
        "model_config": config_to_plain(config.model),
        "scaler": _plain_value(deepcopy(scaler or {})),
    }
    torch.save(checkpoint, path)


def load_teacher_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)
