from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unskild_smolvla.checkpoint_io import resolve_checkpoint
from unskild_smolvla.config import DEFAULT_INTERFACE_SCHEMA_PATH
from unskild_smolvla.config import DEFAULT_MODEL_CONFIG_PATH
from unskild_smolvla.config import load_interface_schema
from unskild_smolvla.config import load_model_config
from unskild_smolvla.inference import load_model
from unskild_smolvla.inference import validate_action
from unskild_smolvla.inference import validate_observation
from unskild_smolvla.model import SmolVLAAdapterModel
from unskild_smolvla.model import SmolVLAAdapterModelConfig


def _valid_observation() -> dict:
    return {
        "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "hand_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "state": np.zeros((8,), dtype=np.float32),
        "prompt": "pick up the cup",
    }


def _write_dummy_checkpoint(path: Path) -> None:
    model_cfg = SmolVLAAdapterModelConfig.from_dict(load_model_config(DEFAULT_MODEL_CONFIG_PATH))
    model = SmolVLAAdapterModel(model_cfg)
    torch.save({"state_dict": model.state_dict()}, path)


def test_observation_schema_exact_match_passes() -> None:
    schema = load_interface_schema(DEFAULT_INTERFACE_SCHEMA_PATH)
    validate_observation(_valid_observation(), schema)


def test_observation_missing_key_fails() -> None:
    schema = load_interface_schema(DEFAULT_INTERFACE_SCHEMA_PATH)
    obs = _valid_observation()
    obs.pop("state")
    with pytest.raises(ValueError, match="Observation key order mismatch|Missing required observation key"):
        validate_observation(obs, schema)


def test_observation_extra_key_fails() -> None:
    schema = load_interface_schema(DEFAULT_INTERFACE_SCHEMA_PATH)
    obs = _valid_observation()
    obs["extra"] = 1
    with pytest.raises(ValueError, match="Observation key order mismatch|Observation keys mismatch"):
        validate_observation(obs, schema)


def test_observation_dtype_mismatch_fails() -> None:
    schema = load_interface_schema(DEFAULT_INTERFACE_SCHEMA_PATH)
    obs = _valid_observation()
    obs["state"] = obs["state"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_observation(obs, schema)


def test_observation_shape_mismatch_fails() -> None:
    schema = load_interface_schema(DEFAULT_INTERFACE_SCHEMA_PATH)
    obs = _valid_observation()
    obs["head_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_observation(obs, schema)


def test_action_schema_mismatch_fails() -> None:
    schema = load_interface_schema(DEFAULT_INTERFACE_SCHEMA_PATH)
    bad_output = {"actions": np.zeros((15, 10), dtype=np.float32)}
    with pytest.raises(ValueError, match="Action dim mismatch"):
        validate_action(bad_output, schema)


def test_checkpoint_resolver_local_path_passes(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _write_dummy_checkpoint(ckpt)
    resolved = resolve_checkpoint(str(ckpt))
    assert resolved == ckpt.resolve()


def test_checkpoint_resolver_sha_mismatch_fails(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _write_dummy_checkpoint(ckpt)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_checkpoint(str(ckpt), expected_sha256="0" * 64)


def test_checkpoint_resolver_r2_uri_passes_with_mocked_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloaded_bytes = b"dummy-weights"

    class _FakeClient:
        def download_file(self, bucket: str, key: str, dest: str) -> None:
            assert bucket == "team-bucket"
            assert key == "checkpoints/model.pt"
            Path(dest).write_bytes(downloaded_bytes)

    class _FakeSession:
        def client(self, *_args, **_kwargs):
            return _FakeClient()

    fake_boto3 = SimpleNamespace(session=SimpleNamespace(Session=_FakeSession))

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")

    resolved = resolve_checkpoint(
        "r2://team-bucket/checkpoints/model.pt",
        cache_dir=tmp_path / "cache",
    )
    assert resolved.exists()
    assert resolved.read_bytes() == downloaded_bytes


def test_dummy_inference_smoke(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _write_dummy_checkpoint(ckpt)
    sha256 = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    policy = load_model(
        str(ckpt),
        device="cpu",
        checkpoint_sha256=sha256,
        model_config_path=DEFAULT_MODEL_CONFIG_PATH,
        interface_schema_path=DEFAULT_INTERFACE_SCHEMA_PATH,
    )
    output = policy.infer(_valid_observation())
    actions = output["actions"]

    cfg = load_model_config(DEFAULT_MODEL_CONFIG_PATH)
    assert actions.shape == (int(cfg["action_horizon"]), int(cfg["action_dim"]))
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()
