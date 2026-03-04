# Unskild SmolVLA Training Walkthrough (HF Dataset -> `model_final.pt`)

This walkthrough shows how to train the `unskild_smolvla` adapter on a Hugging Face dataset and create a submission checkpoint.

## 1. Prerequisites

- Python 3.11
- CUDA GPU recommended
- This repo checked out

Install runtime + adapter dependencies:

```bash
pip install -r requirements.txt
```

Install Hugging Face dataset tooling for training:

```bash
pip install datasets
```

Optional HF login for private datasets:

```bash
huggingface-cli login
```

## 2. Dataset Contract

The training script expects one sample to provide:

- `head_rgb`: image `(H, W, 3)` (or `3, H, W`), uint8 or float image
- `hand_rgb`: image `(H, W, 3)` (or `3, H, W`)
- `state`: float vector `(8,)`
- `prompt`: language string
- `actions`: float array `(T, >=11)`

You can remap keys using CLI arguments if your dataset uses different field names.

For your dataset:

- Hugging Face dataset: `airoa-org/airoa-moma`
- URL: `https://huggingface.co/datasets/airoa-org/airoa-moma/tree/main`

## 3. Train the Adapter

From repo root:

```bash
PYTHONPATH=src:packages/policy-client/src \
python -m unskild_smolvla.train \
  --dataset-id airoa-org/airoa-moma \
  --dataset-config <optional_config_name> \
  --split train \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --device cuda \
  --output ./checkpoints/unskild_smolvla/model_final.pt
```

If you need to list available configs/splits first:

```bash
python - <<'PY'
from datasets import get_dataset_config_names, load_dataset_builder
name = "airoa-org/airoa-moma"
print("configs:", get_dataset_config_names(name))
for cfg in get_dataset_config_names(name):
    b = load_dataset_builder(name, cfg)
    print(cfg, "splits:", list((b.info.splits or {}).keys()))
PY
```

If your dataset keys are different:

```bash
PYTHONPATH=src:packages/policy-client/src \
python -m unskild_smolvla.train \
  --dataset-id <dataset> \
  --state-key observation_state \
  --prompt-key instruction \
  --head-rgb-key camera_head \
  --hand-rgb-key camera_hand \
  --actions-key action \
  --output ./checkpoints/unskild_smolvla/model_final.pt
```

The script writes:

- checkpoint file with `state_dict`
- model metadata
- printed SHA256 checksum

## 4. Verify Checkpoint is Loadable

Run smoke test (requires `torch`):

```bash
PYTHONPATH=src:packages/policy-client/src \
pytest -q src/unskild_smolvla/test_smoke.py
```

## 5. Freeze Submission Config

Edit:

- `submissions/icra2026_unskild_smolvla.yaml`

Update at minimum:

- `checkpoint_uri`
- `checkpoint_sha256`
- `device_default`

## 6. Upload Weights to R2

Upload only the checkpoint file:

- `model_final.pt`

Then set the R2 URI in submission YAML.

## 7. Run Policy Server with Frozen Config

```bash
export POLICY_NAME=unskild_smolvla
export POLICY_SUBMISSION_CONFIG=/workspace/submissions/icra2026_unskild_smolvla.yaml
./RUN-DOCKER-CONTAINER.sh up
```

Local offline fallback (manual file placement in mounted checkpoint path):

```bash
export POLICY_NAME=unskild_smolvla
export POLICY_SUBMISSION_CONFIG=/workspace/submissions/icra2026_unskild_smolvla.yaml
export POLICY_CHECKPOINT_URI=/policy_checkpoint/model_final.pt
./RUN-DOCKER-CONTAINER.sh up
```

## 8. Expected Checkpoint Format

`unskild_smolvla.load_model()` accepts:

- a `.pt` file containing either:
  - `{"state_dict": ...}`
  - or direct state dict mapping param name -> tensor

Weights are loaded with `strict=True`, so missing/unexpected keys fail immediately.
