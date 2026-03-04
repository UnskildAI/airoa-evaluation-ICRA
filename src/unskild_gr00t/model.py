from __future__ import annotations

from pathlib import Path
from typing import Any


def load_model(checkpoint_path: str, device: str) -> Any:
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required for unskild_gr00t policy.")

    resolved_checkpoint = Path(checkpoint_path).expanduser()
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"GR00T checkpoint path not found: {resolved_checkpoint}")

    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy import Gr00tPolicy
    except ImportError as exc:
        raise ImportError(
            "GR00T dependency is missing. Install NVIDIA Isaac-GR00T from source "
            "(git clone + uv sync + uv pip install -e .) and retry."
        ) from exc

    policy = Gr00tPolicy(
        model_path=str(resolved_checkpoint),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device=device,
        strict=True,
    )
    return policy
