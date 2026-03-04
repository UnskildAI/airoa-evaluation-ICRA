# Unskild GR00T Walkthrough (AIRoA HF Dataset -> Train -> Inference)

This walkthrough is GR00T-only and shows how to train on the AIRoA Hugging Face dataset and run the trained checkpoint in this evaluation runtime.

## 1. Dataset

- Dataset: `airoa-org/airoa-moma`
- URL: `https://huggingface.co/datasets/airoa-org/airoa-moma`

The GR00T training pipeline expects GR00T-flavored LeRobot v2 data (`meta/modality.json` required).

## 2. Prepare Data Locally

Accept dataset access on Hugging Face first, then clone with Git LFS:

```bash
huggingface-cli login
git lfs install

git clone https://huggingface.co/datasets/airoa-org/airoa-moma /data/airoa-moma
cd /data/airoa-moma
git lfs pull
```

Quick structure check:

```bash
python - <<'PY'
from pathlib import Path
root = Path("/data/airoa-moma")
required = [
    root / "meta" / "modality.json",
    root / "meta" / "episodes.jsonl",
    root / "meta" / "tasks.jsonl",
    root / "meta" / "info.json",
]
missing = [str(p) for p in required if not p.exists()]
print("missing:", missing if missing else "none")
print("data_dir_exists:", (root / "data").exists())
print("videos_dir_exists:", (root / "videos").exists())
PY
```

## 3. Set Up Isaac-GR00T Training Environment

Use NVIDIA's source-based install flow:

```bash
conda create -n gr00t python=3.10 -y
conda activate gr00t

git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

uv sync --python 3.10
uv pip install -e .
```

## 4. Build Modality Config for NEW_EMBODIMENT

Inspect modality keys in AIRoA dataset:

```bash
python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("/data/airoa-moma/meta/modality.json").read_text())
for k in ("video", "state", "action", "annotation"):
    print(k, "->", list((cfg.get(k) or {}).keys()))
PY
```

Create a modality config file in Isaac-GR00T repo (replace keys with values from your dataset):

```python
# examples/AIROA/airoa_config.py
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig
from gr00t.data.types import ActionFormat
from gr00t.data.types import ActionRepresentation
from gr00t.data.types import ActionType
from gr00t.data.types import ModalityConfig

airoa_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "hand"],  # replace with modality.json video keys
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["robot_state"],  # replace with modality.json state keys
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["robot_action"],  # replace with modality.json action keys
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(airoa_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

## 5. Fine-Tune GR00T on AIRoA Dataset

Run from inside Isaac-GR00T repo:

```bash
export NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0 uv run python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /data/airoa-moma \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/AIROA/airoa_config.py \
  --num-gpus ${NUM_GPUS} \
  --output-dir /tmp/airoa_gr00t \
  --save-total-limit 5 \
  --save-steps 2000 \
  --max-steps 20000 \
  --global-batch-size 32 \
  --dataloader-num-workers 4
```

Optional open-loop eval:

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /data/airoa-moma \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/airoa_gr00t/checkpoint-20000 \
  --traj-ids 0
```

## 6. Run Trained Checkpoint in This Repo

From this repo root:

```bash
pip install -r requirements.txt

PYTHONPATH=src:packages/policy-client/src \
python run_evaluation.py \
  --policy unskild_gr00t \
  --checkpoint-path /tmp/airoa_gr00t/checkpoint-20000 \
  --device cuda
```

CPU fallback:

```bash
PYTHONPATH=src:packages/policy-client/src \
python run_evaluation.py \
  --policy unskild_gr00t \
  --checkpoint-path /tmp/airoa_gr00t/checkpoint-20000 \
  --device cpu
```

## 7. Docker Inference (This Repo)

Optional: install GR00T from source during container build:

```bash
export INSTALL_GR00T_FROM_SOURCE=1
export GR00T_REPO_URL=https://github.com/NVIDIA/Isaac-GR00T.git
export GR00T_REF=main
```

Run server:

```bash
export POLICY_NAME=unskild_gr00t
export POLICY_CHECKPOINT_URI=/policy_checkpoint/gr00t_model
export POLICY_DEVICE=cpu
./RUN-DOCKER-CONTAINER.sh up
```

## 8. Adapter Runtime Contract

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

## 9. Fail-Fast Behavior

The GR00T adapter fails immediately when:

- `--checkpoint-path` (or resolved checkpoint path) is missing
- GR00T Python dependency is missing from the runtime environment
- observation fields are missing or malformed for conversion
- GR00T action shape is invalid for evaluation (`D != 11`)
- GR00T action contains non-finite values
