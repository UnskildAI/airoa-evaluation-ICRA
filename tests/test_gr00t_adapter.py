from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from unskild_gr00t.inference import UnskildGR00TPolicy


def _install_fake_gr00t(monkeypatch: pytest.MonkeyPatch) -> None:
    gr00t_module = ModuleType("gr00t")
    gr00t_module.__path__ = []  # type: ignore[attr-defined]

    policy_module = ModuleType("gr00t.policy")

    data_module = ModuleType("gr00t.data")
    data_module.__path__ = []  # type: ignore[attr-defined]
    embodiment_tags_module = ModuleType("gr00t.data.embodiment_tags")

    class _EmbodimentTag:
        NEW_EMBODIMENT = "NEW_EMBODIMENT"

    class _FakeGr00tPolicy:
        def __init__(
            self,
            *,
            model_path: str,
            embodiment_tag: str,
            device: str,
            strict: bool,
        ) -> None:
            assert model_path
            assert embodiment_tag == _EmbodimentTag.NEW_EMBODIMENT
            assert device == "cpu"
            assert strict is True

        def get_action(self, obs: dict) -> dict:
            assert obs["video"]["head"].shape == (1, 1, 480, 640, 3)
            assert obs["video"]["hand"].shape == (1, 1, 480, 640, 3)
            assert obs["state"]["robot_state"].shape == (1, 1, 8)
            assert obs["language"]["task"] == [["pick up the cup"]]
            action = {"joint_actions": np.ones((1, 1, 11), dtype=np.float32)}
            info = {"latency_ms": 1.0}
            return action, info

    policy_module.Gr00tPolicy = _FakeGr00tPolicy
    embodiment_tags_module.EmbodimentTag = _EmbodimentTag

    gr00t_module.policy = policy_module  # type: ignore[attr-defined]
    gr00t_module.data = data_module  # type: ignore[attr-defined]
    data_module.embodiment_tags = embodiment_tags_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gr00t", gr00t_module)
    monkeypatch.setitem(sys.modules, "gr00t.policy", policy_module)
    monkeypatch.setitem(sys.modules, "gr00t.data", data_module)
    monkeypatch.setitem(sys.modules, "gr00t.data.embodiment_tags", embodiment_tags_module)


def _dummy_observation() -> dict:
    return {
        "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "hand_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "state": np.zeros((8,), dtype=np.float32),
        "prompt": "pick up the cup",
    }


def test_gr00t_policy_predict_smoke(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gr00t(monkeypatch)
    checkpoint_dir = tmp_path / "gr00t_model"
    checkpoint_dir.mkdir()

    policy = UnskildGR00TPolicy(str(checkpoint_dir), device="cpu")
    output = policy.predict(_dummy_observation())
    infer_output = policy.infer(_dummy_observation())

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 11)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert "actions" in infer_output
    assert infer_output["actions"].shape == (1, 11)


def test_gr00t_policy_missing_checkpoint_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gr00t(monkeypatch)
    missing_checkpoint = tmp_path / "missing_gr00t_model"
    with pytest.raises(FileNotFoundError, match="GR00T checkpoint path not found"):
        UnskildGR00TPolicy(str(missing_checkpoint), device="cpu")
