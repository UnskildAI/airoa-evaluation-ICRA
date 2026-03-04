from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from unskild_smolvla.config import DEFAULT_MODEL_CONFIG_PATH
from unskild_smolvla.config import load_model_config
from unskild_smolvla.model import SmolVLAAdapterModel
from unskild_smolvla.model import SmolVLAAdapterModelConfig


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _as_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Image must be 3D, got shape={arr.shape}")

    # CHW -> HWC if needed.
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] != 3:
        raise ValueError(f"Image must have 3 channels, got shape={arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.max(arr)) if arr.size else 1.0
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class HFDatasetAdapter(Dataset):
    def __init__(
        self,
        hf_dataset: Any,
        *,
        model_cfg: SmolVLAAdapterModelConfig,
        state_key: str,
        prompt_key: str,
        head_rgb_key: str,
        hand_rgb_key: str,
        actions_key: str,
    ) -> None:
        self.ds = hf_dataset
        self.model_cfg = model_cfg
        self.state_key = state_key
        self.prompt_key = prompt_key
        self.head_rgb_key = head_rgb_key
        self.hand_rgb_key = hand_rgb_key
        self.actions_key = actions_key

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.ds[idx]
        head_rgb = _as_hwc_uint8(row[self.head_rgb_key])
        hand_rgb = _as_hwc_uint8(row[self.hand_rgb_key])

        state = np.asarray(row[self.state_key], dtype=np.float32).reshape(-1)
        if state.shape[0] != self.model_cfg.state_dim:
            raise ValueError(
                f"State size mismatch at idx={idx}: expected={self.model_cfg.state_dim}, got={state.shape[0]}"
            )

        prompt = str(row[self.prompt_key])
        actions = np.asarray(row[self.actions_key], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Actions must be 2D at idx={idx}, got shape={actions.shape}")
        if actions.shape[1] < self.model_cfg.action_dim:
            raise ValueError(
                f"Action dim too small at idx={idx}: expected >= {self.model_cfg.action_dim}, got {actions.shape[1]}"
            )

        target = actions[: self.model_cfg.action_horizon, : self.model_cfg.action_dim]
        if target.shape[0] == 0:
            raise ValueError(f"No action steps found at idx={idx}")
        if target.shape[0] < self.model_cfg.action_horizon:
            # Pad by repeating last step to fixed horizon.
            pad_steps = self.model_cfg.action_horizon - target.shape[0]
            last = target[-1:, :]
            target = np.concatenate([target, np.repeat(last, pad_steps, axis=0)], axis=0)

        return {
            "head_rgb": head_rgb,
            "hand_rgb": hand_rgb,
            "state": state,
            "prompt": prompt,
            "target_actions": target.astype(np.float32, copy=False),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "head_rgb": np.stack([b["head_rgb"] for b in batch], axis=0),
        "hand_rgb": np.stack([b["hand_rgb"] for b in batch], axis=0),
        "state": np.stack([b["state"] for b in batch], axis=0),
        "prompt": [b["prompt"] for b in batch],
        "target_actions": np.stack([b["target_actions"] for b in batch], axis=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Unskild SmolVLA adapter on a Hugging Face dataset and export model weights."
    )
    parser.add_argument("--dataset-id", required=True, help="Hugging Face dataset id, e.g. org/hsr-vla")
    parser.add_argument("--dataset-config", default=None, help="Optional HF dataset config name")
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples for quick runs (0 = full split)")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH), help="Path to model_config.yaml")
    parser.add_argument("--output", required=True, help="Output checkpoint path (.pt)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Dataset key mapping
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--head-rgb-key", default="head_rgb")
    parser.add_argument("--hand-rgb-key", default="hand_rgb")
    parser.add_argument("--actions-key", default="actions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Hugging Face datasets package is required for training. "
            "Install with: pip install datasets"
        ) from exc

    model_cfg_dict = load_model_config(args.model_config)
    model_cfg = SmolVLAAdapterModelConfig.from_dict(model_cfg_dict)

    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    train_ds = HFDatasetAdapter(
        ds,
        model_cfg=model_cfg,
        state_key=args.state_key,
        prompt_key=args.prompt_key,
        head_rgb_key=args.head_rgb_key,
        hand_rgb_key=args.hand_rgb_key,
        actions_key=args.actions_key,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )

    device = torch.device(args.device)
    model = SmolVLAAdapterModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        num_steps = 0
        for batch in train_loader:
            head_rgb = torch.from_numpy(batch["head_rgb"]).to(device)
            hand_rgb = torch.from_numpy(batch["hand_rgb"]).to(device)
            state = torch.from_numpy(batch["state"]).to(device)
            target = torch.from_numpy(batch["target_actions"]).to(device)
            prompts = batch["prompt"]

            pred = model(
                head_rgb=head_rgb,
                hand_rgb=hand_rgb,
                state=state,
                prompts=prompts,
            )
            loss = loss_fn(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_steps += 1

        mean_loss = running_loss / max(num_steps, 1)
        print(f"[epoch {epoch}/{args.epochs}] mean_loss={mean_loss:.6f}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "model_config": model_cfg_dict,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "split": args.split,
    }
    torch.save(payload, output_path)

    sha256 = _sha256_file(output_path)
    print(f"Saved checkpoint: {output_path}")
    print(f"SHA256: {sha256}")


if __name__ == "__main__":
    main()
