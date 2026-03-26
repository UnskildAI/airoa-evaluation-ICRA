# REPRODUCTION GUIDE

To evaluate the Unskild AI submission on a Blackwell (RTX 50-series) GPU, please follow these exact steps. This ensuring the environment is correctly synchronized and architecture-compatible.

### 1. Prepare the Environment
```bash
# Set the checkpoint path provided in your evaluation environment
export POLICY_CHECKPOINT_PATH="/policy_checkpoint/gr00t_model"

# Model and Build configuration
export POLICY_NAME="unskild_gr00t"
export INSTALL_GR00T_FROM_SOURCE=1
export GR00T_REPO_URL="https://github.com/NVIDIA/Isaac-GR00T.git"
export GR00T_REF="main"
```

### 2. Launch the Policy Server & Client
```bash
# Clean up any existing containers and start a fresh build
./RUN-DOCKER-CONTAINER.sh down
./RUN-DOCKER-CONTAINER.sh up
```

### 3. Run Inference
```bash
# Open the HSR client shell
./RUN-DOCKER-CONTAINER.sh shell

# Inside the shell, run the launch command
roslaunch hsr_policy_client hsr_policy_client.launch
```

---

### Technical Notes for Organizers
- **Blackwell Support**: The `policy_server` image uses `nvidia/cuda:12.6.3-devel-ubuntu22.04` to support SM 10.0+ (Blackwell) hardware.
- **uv Manager**: We have resolved prefix synchronization issues by explicitly setting `UV_PYTHON=python3.11` and ensuring the system `python3.11-dev` headers are present in the server image.
- **Client Docker**: As requested, the `client/Dockerfile` remains in its original standard state without any modifications.
