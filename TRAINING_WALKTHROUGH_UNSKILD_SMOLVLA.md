# Unskild GR00T Walkthrough (Training + Inference)

This walkthrough is GR00T-only and covers how to use the AIRoA dataset for GR00T training, then run inference in this evaluation runtime.

## 1. Dataset Location

- Hugging Face dataset: `airoa-org/airoa-moma`
- URL: `https://huggingface.co/datasets/airoa-org/airoa-moma`

## 2. Prerequisites

Training environment (official GR00T recommendation):

- Conda
- Python 3.10
- CUDA-capable NVIDIA GPU machine

Inference/evaluation environment (this repo):

- Python 3.11 for this repository runtime
- This repo checked out

Optional Hugging Face login for private dataset access:

```bash
huggingface-cli login
```

## 3. Install Isaac-GR00T from Source

`isaac-gr00t` is not installed as a simple PyPI package in this workflow.
Use the official source-based install:

```bash
conda create -n gr00t python=3.10 -y
conda activate gr00t

git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

# Recommended by upstream:
uv sync --python 3.10
uv pip install -e .
```

Alternative (if using pip workflow in the same environment):

```bash
pip install -e .
```

If you are using LeRobot with GR00T extras, upstream docs also describe:

```bash
pip install -e ".[groot,dev,test]"
```

## 4. GR00T Training (Outside Adapter)

This repo does not implement GR00T training code.
Train/fine-tune with the official GR00T pipeline, then export a GR00T model directory/checkpoint path.

Use this dataset as your training source:

- `airoa-org/airoa-moma`

Make sure your trained artifact is accessible as a local path for inference, for example:

- `/policy_checkpoint/gr00t_model`

## 5. GR00T Inference in This Repo

From this repo root, install runtime deps for the evaluation server:

```bash
pip install -r requirements.txt
```

Run directly from repo root:

```bash
PYTHONPATH=src:packages/policy-client/src \
python run_evaluation.py \
  --policy unskild_gr00t \
  --checkpoint-path /policy_checkpoint/gr00t_model \
  --device cuda
```

CPU fallback:

```bash
PYTHONPATH=src:packages/policy-client/src \
python run_evaluation.py \
  --policy unskild_gr00t \
  --checkpoint-path /policy_checkpoint/gr00t_model \
  --device cpu
```

## 6. Docker Inference

Default container build keeps GR00T source install disabled (for CPU-safe builds).
To enable source install during image build:

```bash
export INSTALL_GR00T_FROM_SOURCE=1
export GR00T_REPO_URL=https://github.com/NVIDIA/Isaac-GR00T.git
export GR00T_REF=main
```

```bash
export POLICY_NAME=unskild_gr00t
export POLICY_CHECKPOINT_URI=/policy_checkpoint/gr00t_model
export POLICY_DEVICE=cpu
./RUN-DOCKER-CONTAINER.sh up
```

## 7. Runtime Contract (GR00T Adapter)

Input observation expected by server:

- `head_rgb`: `(H, W, 3)`
- `hand_rgb`: `(H, W, 3)`
- `state`: `(8,)`
- `prompt`: `str`

Adapter conversion to GR00T runtime:

- images cast to uint8 RGB
- batch/time dimensions added: `B=1`, `T=1`
- state cast to float32
- language packed as `[[prompt]]`

Output returned to evaluation client:

- `actions`: numpy `float32`, shape `(T, 11)`

## 8. Fail-Fast Behavior

The GR00T adapter fails immediately when:

- `--checkpoint-path` (or resolved checkpoint path) is missing
- GR00T Python dependency is missing from the runtime environment
- observation fields are missing or malformed for conversion
- GR00T action shape is invalid for evaluation (`D != 11`)
- GR00T action contains non-finite values
