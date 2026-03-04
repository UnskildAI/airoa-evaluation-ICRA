from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from policy_client import base_policy as _base_policy
import torch

from unskild_smolvla.checkpoint_io import resolve_checkpoint
from unskild_smolvla.config import load_interface_schema
from unskild_smolvla.config import load_model_config
from unskild_smolvla.model import SmolVLAAdapterModel
from unskild_smolvla.model import SmolVLAAdapterModelConfig


BasePolicy = _base_policy.BasePolicy
_ACTIVE_POLICY: UnskildSmolVLAPolicy | None = None


def _assert_array_shape(arr: np.ndarray, expected_shape: list[Any], field_name: str) -> None:
    if len(expected_shape) != arr.ndim:
        raise ValueError(
            f"Field '{field_name}' rank mismatch: expected ndim={len(expected_shape)}, got {arr.ndim}"
        )
    for i, expected in enumerate(expected_shape):
        if isinstance(expected, int) and arr.shape[i] != expected:
            raise ValueError(
                f"Field '{field_name}' shape mismatch at dim={i}: expected={expected}, got={arr.shape[i]}"
            )


def validate_observation(observation: dict[str, Any], schema: dict[str, Any]) -> None:
    obs_schema = schema["observation"]
    ordered_keys = list(obs_schema["ordered_keys"])
    strict_order = bool(obs_schema.get("strict_order", True))
    strict_keys = bool(obs_schema.get("strict_keys", True))
    fields = obs_schema["fields"]

    given_keys = list(observation.keys())
    if strict_order and given_keys != ordered_keys:
        raise ValueError(
            f"Observation key order mismatch. expected={ordered_keys}, got={given_keys}"
        )

    expected_set = set(ordered_keys)
    actual_set = set(given_keys)
    if strict_keys and actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(f"Observation keys mismatch. missing={missing}, extra={extra}")

    for key in ordered_keys:
        field = fields[key]
        required = bool(field.get("required", True))
        if required and key not in observation:
            raise ValueError(f"Missing required observation key: {key}")
        if key not in observation:
            continue

        value = observation[key]
        dtype_name = field["dtype"]
        if dtype_name == "str":
            if not isinstance(value, str):
                raise ValueError(f"Field '{key}' must be str, got {type(value).__name__}")
            continue

        arr = np.asarray(value)
        expected_dtype = np.dtype(dtype_name)
        if arr.dtype != expected_dtype:
            raise ValueError(
                f"Field '{key}' dtype mismatch: expected={expected_dtype}, got={arr.dtype}"
            )
        expected_shape = field.get("shape")
        if expected_shape is not None:
            _assert_array_shape(arr, list(expected_shape), key)
        if bool(field.get("finite", False)) and not np.isfinite(arr).all():
            raise ValueError(f"Field '{key}' contains non-finite values.")


def validate_action(output: dict[str, Any], schema: dict[str, Any]) -> None:
    action_schema = schema["action"]
    action_key = str(action_schema.get("key", "actions"))
    strict_keys = bool(action_schema.get("strict_keys", True))

    keys = list(output.keys())
    if strict_keys and keys != [action_key]:
        raise ValueError(f"Action output keys mismatch. expected=['{action_key}'], got={keys}")
    if action_key not in output:
        raise ValueError(f"Missing required output action key: {action_key}")

    arr = np.asarray(output[action_key])
    expected_dtype = np.dtype(action_schema["dtype"])
    if arr.dtype != expected_dtype:
        raise ValueError(f"Action dtype mismatch: expected={expected_dtype}, got={arr.dtype}")

    shape = list(action_schema["shape"])
    if arr.ndim != len(shape):
        raise ValueError(f"Action rank mismatch: expected ndim={len(shape)}, got={arr.ndim}")
    if isinstance(shape[1], int) and arr.shape[1] != shape[1]:
        raise ValueError(f"Action dim mismatch: expected={shape[1]}, got={arr.shape[1]}")

    min_horizon = int(action_schema.get("min_horizon", 1))
    if arr.shape[0] < min_horizon:
        raise ValueError(f"Action horizon too short: expected >= {min_horizon}, got={arr.shape[0]}")

    if bool(action_schema.get("finite", False)) and not np.isfinite(arr).all():
        raise ValueError("Action output contains non-finite values.")


class UnskildSmolVLAPolicy(BasePolicy):
    def __init__(
        self,
        model: SmolVLAAdapterModel,
        *,
        schema: dict[str, Any],
        device: str = "cpu",
        default_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._schema = schema
        self._device = torch.device(device)
        self._default_prompt = default_prompt
        self._action_key = str(schema["action"].get("key", "actions"))

    def predict(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        obs = dict(observation)
        if "prompt" not in obs and self._default_prompt is not None:
            obs["prompt"] = self._default_prompt

        validate_observation(obs, self._schema)

        head_rgb = np.asarray(obs["head_rgb"], dtype=np.uint8)
        hand_rgb = np.asarray(obs["hand_rgb"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        prompt = str(obs["prompt"])

        with torch.no_grad():
            action = self._model(
                head_rgb=torch.from_numpy(head_rgb).to(self._device)[None, ...],
                hand_rgb=torch.from_numpy(hand_rgb).to(self._device)[None, ...],
                state=torch.from_numpy(state).to(self._device)[None, ...],
                prompts=[prompt],
            )[0]

        output = {
            self._action_key: action.detach().cpu().numpy().astype(np.float32, copy=False),
        }
        validate_action(output, self._schema)
        return output

    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        return self.predict(obs)


def load_model(
    checkpoint_path: str,
    *,
    device: str = "cpu",
    checkpoint_sha256: str | None = None,
    model_config_path: str | Path | None = None,
    interface_schema_path: str | Path | None = None,
    default_prompt: str | None = None,
) -> UnskildSmolVLAPolicy:
    """Load model from checkpoint and set active policy instance."""
    config_dict = load_model_config(model_config_path)
    model_config = SmolVLAAdapterModelConfig.from_dict(config_dict)

    resolved_checkpoint = resolve_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
    )
    checkpoint = torch.load(resolved_checkpoint, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {resolved_checkpoint}: expected dict or dict with 'state_dict'."
        )

    model = SmolVLAAdapterModel(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    schema = load_interface_schema(interface_schema_path)
    policy = UnskildSmolVLAPolicy(
        model,
        schema=schema,
        device=device,
        default_prompt=default_prompt,
    )

    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy
    return policy


def predict(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run prediction with the active policy instance loaded by load_model()."""
    if _ACTIVE_POLICY is None:
        raise RuntimeError("No active policy loaded. Call load_model(checkpoint_path=...) first.")
    return _ACTIVE_POLICY.predict(observation)
