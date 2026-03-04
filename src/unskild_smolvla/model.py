from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SmolVLAAdapterModelConfig:
    state_dim: int
    prompt_embedding_dim: int
    hidden_dims: Sequence[int]
    action_horizon: int
    action_dim: int
    activation: str = "gelu"
    normalize_images: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> SmolVLAAdapterModelConfig:
        return cls(
            state_dim=int(data["state_dim"]),
            prompt_embedding_dim=int(data["prompt_embedding_dim"]),
            hidden_dims=[int(v) for v in data["hidden_dims"]],
            action_horizon=int(data["action_horizon"]),
            action_dim=int(data["action_dim"]),
            activation=str(data.get("activation", "gelu")),
            normalize_images=bool(data.get("normalize_images", True)),
        )


class SmolVLAAdapterModel(nn.Module):
    """Small deterministic adapter model for HSR websocket inference."""

    def __init__(self, config: SmolVLAAdapterModelConfig) -> None:
        super().__init__()
        self.config = config

        image_feature_dim = 12  # mean/std for RGB from head + hand
        input_dim = config.state_dim + image_feature_dim + config.prompt_embedding_dim

        activation: nn.Module
        if config.activation.lower() == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.GELU()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(activation)
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, config.action_horizon * config.action_dim))
        self.mlp = nn.Sequential(*layers)

    def _prompt_embedding(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        dim = self.config.prompt_embedding_dim
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            repeat = ((dim + len(digest) - 1) // len(digest)) + 1
            raw = np.frombuffer(digest * repeat, dtype=np.uint8)[:dim].astype(np.float32)
            rows.append((raw / 127.5) - 1.0)
        return torch.from_numpy(np.stack(rows, axis=0)).to(device=device)

    def _image_stats(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B,H,W,C], got shape={tuple(image.shape)}")
        image = image.float()
        if self.config.normalize_images:
            image = image / 255.0
        mean = image.mean(dim=(1, 2))
        std = image.std(dim=(1, 2), unbiased=False)
        return torch.cat([mean, std], dim=-1)

    def forward(
        self,
        *,
        head_rgb: torch.Tensor,
        hand_rgb: torch.Tensor,
        state: torch.Tensor,
        prompts: list[str],
    ) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError(f"Expected state shape [B,{self.config.state_dim}], got {tuple(state.shape)}")
        if state.shape[1] != self.config.state_dim:
            raise ValueError(f"state dim mismatch: expected {self.config.state_dim}, got {state.shape[1]}")

        batch_size = state.shape[0]
        if len(prompts) != batch_size:
            raise ValueError(f"Prompt count mismatch: expected {batch_size}, got {len(prompts)}")

        head_stats = self._image_stats(head_rgb)
        hand_stats = self._image_stats(hand_rgb)
        prompt_embed = self._prompt_embedding(prompts, device=state.device)

        features = torch.cat([state.float(), head_stats, hand_stats, prompt_embed], dim=-1)
        output = self.mlp(features)
        return output.view(batch_size, self.config.action_horizon, self.config.action_dim)
