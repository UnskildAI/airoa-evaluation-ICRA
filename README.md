# airoa-evaluation-ICRA

Participant evaluation runtime for ICRA 2026 VLA Workshop Competition.

## 1. Editable Scope

Main implementation targets:

- `server/`
- `src/`

## 2. Host Requirements

- Linux
- Docker Engine
- Docker Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit

```bash
docker --version
docker compose version
nvidia-smi
```

Verified environment (2026-02-20):

- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 5070 Ti
- NVIDIA driver: 580.126.09
- Docker: 29.0.1
- Docker Compose: v2.40.3

## 3. Implementation Workflow

1. Create a feature branch from the prepared base branch.
2. Add your model repository under `src/`.
3. Update `server/serve_hsr_policy_ws.py` (currently the OpenPI version) and server-side dependencies for your model runtime.
4. Add an adapter in `src/` or `server/`.

Adapter definition:
An adapter is a thin conversion layer that maps between the HSR client contract and your model's native
input/output format. For required fields and shapes, just follow the `WebSocket I/O Contract` section below.

5. Run the test flow and confirm `Action executed.` appears in logs.
6. Submit your branch.

## 4. Required Environment Variables

```bash
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
```

## 5. Sample Openpi Variables
```bash
export POLICY_CHECKPOINT_PATH=/abs/path/to/pi05_hsr_task6891011_level12_v2.5_train_adaptive/pi05_hsr_task6891011_level12_v2.5_train_adaptive_gpu8/200000/
export POLICY_CONFIG_NAME=pi05_hsr_task6891011_level12_v2.5_train_adaptive
```

## 6. Test Flow

Start containers:

```bash
./RUN-DOCKER-CONTAINER.sh up
```

Enter client shell:

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

Run launch inside the container:

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

By default, `test_mode` is `true`. In this mode, the client uses synthetic random observations
(`head_rgb`, `hand_rgb`, and `state`) in an infinite loop and prints language/action logs.

## 7. Unskild SmolVLA Submission Mode

Frozen submission config:

- `submissions/icra2026_unskild_smolvla.yaml`

Local training + weight export walkthrough:

- `TRAINING_WALKTHROUGH_UNSKILD_SMOLVLA.md`

Set environment:

```bash
export POLICY_NAME=unskild_smolvla
export POLICY_SUBMISSION_CONFIG=/workspace/submissions/icra2026_unskild_smolvla.yaml
```

Optional local fallback (restricted network):

```bash
export POLICY_CHECKPOINT_URI=/policy_checkpoint/model_final.pt
export POLICY_DEVICE=cpu
```

Optional R2 credentials (for `r2://` checkpoints):

```bash
export R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
export R2_ACCESS_KEY_ID=<key>
export R2_SECRET_ACCESS_KEY=<secret>
export R2_BUCKET=<bucket>
```

## 8. Running GR00T Policy

Install GR00T from NVIDIA source (not via simple PyPI install):

```bash
conda create -n gr00t python=3.10 -y
conda activate gr00t
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
uv sync --python 3.10
uv pip install -e .
```

Then run the GR00T adapter directly:

```bash
python run_evaluation.py \
  --policy unskild_gr00t \
  --checkpoint-path /policy_checkpoint/gr00t_model \
  --device cuda
```

If `--device` is omitted, the default is `cpu`.

Docker build can optionally install GR00T from source:

```bash
export INSTALL_GR00T_FROM_SOURCE=1
export GR00T_REPO_URL=https://github.com/NVIDIA/Isaac-GR00T.git
export GR00T_REF=main
```

## 9. Clean Install (pip)

For fresh-machine reproducibility:

```bash
pip install -r requirements.txt
```

This keeps existing dependency flow and adds pinned policy-specific requirements from:

- `requirements_unskild_smolvla.txt`

## 10. Logs and Stop

```bash
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh down
```

## 11. WebSocket I/O Contract

Inference request fields:

- `head_rgb`: image array `(H, W, 3)` (current HSR dataset profile: `(480, 640, 3)`)
- `hand_rgb`: image array `(H, W, 3)` (current HSR dataset profile: `(480, 640, 3)`)
- `state`: `(8,)`
- `prompt`: `str`

Inference response field:

- `actions` with shape `(T, 11)`, `T >= 1`

Action order:

- `[arm_lift_joint, arm_flex_joint, arm_roll_joint, wrist_flex_joint, wrist_roll_joint, gripper, head_pan_joint, head_tilt_joint, base_x, base_y, base_t]`

Value requirement:

- finite numeric values only

## 12. Failure Semantics (Fail-Fast)

The `unskild_smolvla` adapter exits with explicit errors when:

- observation keys/order/dtypes/shapes mismatch the locked schema
- action dtypes/shapes violate the locked schema
- checkpoint path/URI is missing or invalid
- checkpoint SHA256 does not match expected hash
