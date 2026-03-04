#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from runtime_core.websocket_policy_server import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve policy as websocket server for HSR client")
    parser.add_argument(
        "--policy",
        default=os.getenv("POLICY_NAME", "openpi"),
        choices=["openpi", "unskild_smolvla", "unskild_gr00t"],
        help="Policy backend to serve.",
    )
    parser.add_argument(
        "--submission-config",
        default=None,
        help="Path to frozen submission config YAML for reproducible runs.",
    )

    # OpenPI arguments
    parser.add_argument("--checkpoint-dir", default=None, help="Path to OpenPI checkpoint directory")
    parser.add_argument("--config-name", default=None, help="OpenPI train config name (e.g. pi05_hsr)")
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help='Optional OpenPI torch device override (e.g. "cuda", "cuda:0", "cpu")',
    )

    # Unskild SmolVLA arguments
    parser.add_argument(
        "--checkpoint-uri",
        default=None,
        help="Checkpoint URI/path for unskild_smolvla policy (supports r2:// and local paths).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint path for unskild_gr00t policy.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default=None,
        help="Expected SHA256 checksum for checkpoint file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device override for unskild policies (e.g. "cuda", "cuda:0", "cpu").',
    )

    # Shared arguments
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--default-prompt", default=None, help="Fallback prompt if prompt key is missing")
    parser.add_argument("--record-dir", default=None, help="Optional directory for policy records")
    return parser.parse_args()


def _load_submission_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Submission config root must be a mapping: {config_path}")
    return cfg


def _build_openpi_policy(args: argparse.Namespace):
    from openpi.policies import policy as policy_lib
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required for policy=openpi")
    if not args.config_name:
        raise ValueError("--config-name is required for policy=openpi")

    checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    config = train_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )

    if args.record_dir:
        policy = policy_lib.PolicyRecorder(policy, args.record_dir)

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "policy_name": "openpi",
            "config_name": args.config_name,
            "checkpoint_dir": checkpoint_dir,
        }
    )
    return policy, metadata


def _build_unskild_policy(args: argparse.Namespace, submission_cfg: dict[str, Any]):
    from unskild_smolvla import inference as unskild_inference

    if args.submission_config and args.checkpoint_sha256:
        raise ValueError(
            "--checkpoint-sha256 override is disabled when --submission-config is set. "
            "Use the SHA in submission YAML."
        )

    checkpoint_uri = args.checkpoint_uri or submission_cfg.get("checkpoint_uri")
    if not checkpoint_uri:
        raise ValueError(
            "Checkpoint URI/path is required for policy=unskild_smolvla. "
            "Set --checkpoint-uri or provide it in --submission-config."
        )

    checkpoint_sha256 = submission_cfg.get("checkpoint_sha256")
    if not args.submission_config and args.checkpoint_sha256:
        checkpoint_sha256 = args.checkpoint_sha256

    # Allow local-path fallback override for restricted network environments.
    if args.submission_config and args.checkpoint_uri:
        checkpoint_uri = args.checkpoint_uri

    if isinstance(checkpoint_sha256, str) and checkpoint_sha256.startswith("REPLACE_"):
        checkpoint_sha256 = None

    model_config_path = submission_cfg.get("model_config_path")
    interface_schema_path = submission_cfg.get("interface_schema_path")
    device = (
        args.device
        or submission_cfg.get("device_default")
        or args.pytorch_device
        or "cpu"
    )

    policy = unskild_inference.load_model(
        checkpoint_uri,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        model_config_path=model_config_path,
        interface_schema_path=interface_schema_path,
        default_prompt=args.default_prompt,
    )

    metadata = {
        "policy_name": "unskild_smolvla",
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256 or "",
        "device": device,
        "submission_config": args.submission_config or "",
        "expected_observation_keys": submission_cfg.get("expected_observation_keys", []),
        "action_dimension": submission_cfg.get("action_dimension"),
        "action_horizon": submission_cfg.get("action_horizon"),
    }
    return policy, metadata


def _build_unskild_gr00t_policy(args: argparse.Namespace, submission_cfg: dict[str, Any]):
    from unskild_gr00t.inference import UnskildGR00TPolicy

    checkpoint_path = (
        args.checkpoint_path
        or submission_cfg.get("checkpoint_path")
        or args.checkpoint_uri
        or submission_cfg.get("checkpoint_uri")
    )
    if not checkpoint_path:
        raise ValueError(
            "Checkpoint path is required for policy=unskild_gr00t. "
            "Set --checkpoint-path (preferred) or provide it in --submission-config."
        )

    device = (
        args.device
        or submission_cfg.get("device_default")
        or args.pytorch_device
        or "cpu"
    )

    policy = UnskildGR00TPolicy(
        checkpoint_path=str(checkpoint_path),
        device=str(device),
    )

    metadata = {
        "policy_name": "unskild_gr00t",
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "submission_config": args.submission_config or "",
        "expected_observation_keys": submission_cfg.get("expected_observation_keys", []),
        "action_dimension": submission_cfg.get("action_dimension"),
        "action_horizon": submission_cfg.get("action_horizon"),
    }
    return policy, metadata


def main() -> None:
    args = parse_args()
    submission_cfg = _load_submission_config(args.submission_config)

    if submission_cfg:
        args.policy = str(submission_cfg.get("policy_name", args.policy))
        if args.device is None:
            args.device = submission_cfg.get("device_default")
        if args.checkpoint_path is None:
            args.checkpoint_path = submission_cfg.get("checkpoint_path")
        if args.checkpoint_uri is None:
            args.checkpoint_uri = submission_cfg.get("checkpoint_uri")

    policy_name = str(args.policy).strip().lower()
    if policy_name == "openpi":
        policy, metadata = _build_openpi_policy(args)
    elif policy_name == "unskild_smolvla":
        policy, metadata = _build_unskild_policy(args, submission_cfg)
    elif policy_name == "unskild_gr00t":
        policy, metadata = _build_unskild_gr00t_policy(args, submission_cfg)
    else:
        raise ValueError(f"Unsupported policy '{policy_name}'")

    if args.record_dir and policy_name != "openpi":
        # Keep behavior parity for non-openpi policies.
        from openpi.policies import policy as policy_lib

        policy = policy_lib.PolicyRecorder(policy, args.record_dir)

    metadata.update(
        {
            "server_host": args.host,
            "server_port": args.port,
        }
    )

    logging.info("Serving policy=%s on %s:%s", policy_name, args.host, args.port)
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
