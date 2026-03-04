from __future__ import annotations

from typing import Any

import numpy as np


EVAL_ACTION_DIM = 11


def _to_uint8_rgb(image: Any, field_name: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Field '{field_name}' must be HxWx3. Got shape={getattr(arr, 'shape', None)}"
        )

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    return np.ascontiguousarray(arr)


def _to_state_vector(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def convert_observation(obs: dict[str, Any]) -> dict[str, Any]:
    required_keys = ("head_rgb", "hand_rgb", "state", "prompt")
    missing = [key for key in required_keys if key not in obs]
    if missing:
        raise ValueError(f"Missing required observation fields for GR00T conversion: {missing}")

    head_rgb = _to_uint8_rgb(obs["head_rgb"], "head_rgb")
    hand_rgb = _to_uint8_rgb(obs["hand_rgb"], "hand_rgb")
    state = _to_state_vector(obs["state"])
    prompt = str(obs["prompt"])

    gr00t_obs = {
        "video": {
            "head": head_rgb[None, None, ...],  # B=1, T=1
            "hand": hand_rgb[None, None, ...],  # B=1, T=1
        },
        "state": {
            "robot_state": state[None, None, ...],  # B=1, T=1
        },
        "language": {
            "task": [[prompt]],
        },
    }
    return gr00t_obs


def convert_gr00t_action(action: dict[str, Any]) -> np.ndarray:
    if not isinstance(action, dict) or not action:
        raise ValueError("GR00T action must be a non-empty mapping.")

    if len(action) != 1:
        raise ValueError(
            f"GR00T action must contain exactly one action tensor. Got keys={list(action.keys())}"
        )

    action_name, action_tensor = next(iter(action.items()))
    arr = np.asarray(action_tensor, dtype=np.float32)

    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"GR00T action batch size must be 1. Got key={action_name}, shape={arr.shape}"
            )
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(
            f"GR00T action tensor must be rank 3 (B,T,D) or rank 2 (T,D). "
            f"Got key={action_name}, shape={arr.shape}"
        )

    if arr.shape[-1] != EVAL_ACTION_DIM:
        raise ValueError(
            f"Action dimension mismatch: expected {EVAL_ACTION_DIM}, got {arr.shape[-1]}"
        )

    if not np.isfinite(arr).all():
        raise ValueError("GR00T action contains non-finite values.")

    return np.ascontiguousarray(arr, dtype=np.float32)
