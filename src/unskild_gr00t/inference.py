from __future__ import annotations

from typing import Any

import numpy as np
from policy_client import base_policy as _base_policy

from unskild_gr00t.model import load_model as _load_gr00t_model
from unskild_gr00t.schema_adapter import convert_gr00t_action
from unskild_gr00t.schema_adapter import convert_observation


BasePolicy = _base_policy.BasePolicy
_ACTIVE_POLICY: UnskildGR00TPolicy | None = None


class UnskildGR00TPolicy(BasePolicy):
    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.policy = _load_gr00t_model(checkpoint_path, device)

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        gr00t_obs = convert_observation(observation)
        result = self.policy.get_action(gr00t_obs)
        action = result[0] if isinstance(result, tuple) else result
        return convert_gr00t_action(action)

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return {"actions": self.predict(obs)}


def load_model(checkpoint_path: str, device: str = "cpu") -> UnskildGR00TPolicy:
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = UnskildGR00TPolicy(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return _ACTIVE_POLICY


def predict(observation: dict[str, Any]) -> np.ndarray:
    if _ACTIVE_POLICY is None:
        raise RuntimeError("No active policy loaded. Call load_model(checkpoint_path=...) first.")
    return _ACTIVE_POLICY.predict(observation)
