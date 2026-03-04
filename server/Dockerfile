FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY requirements.txt requirements_unskild_smolvla.txt ./
COPY packages/policy-client ./packages/policy-client
COPY src/openpi ./src/openpi
COPY src/unskild_gr00t ./src/unskild_gr00t
COPY src/unskild_smolvla ./src/unskild_smolvla
COPY server ./server
COPY runtime_core ./runtime_core

# Create virtual environment and install locked deps including local workspace packages.
RUN uv venv --python 3.11 /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev
RUN uv pip install --python /workspace/.venv/bin/python pytest
RUN uv pip install --python /workspace/.venv/bin/python -r /workspace/requirements_unskild_smolvla.txt

# Optional GR00T source install. Disabled by default to keep CPU-only builds working.
ARG INSTALL_GR00T_FROM_SOURCE=0
ARG GR00T_REPO_URL=https://github.com/NVIDIA/Isaac-GR00T.git
ARG GR00T_REF=main
RUN if [ "${INSTALL_GR00T_FROM_SOURCE}" = "1" ]; then \
      git clone --depth 1 --branch "${GR00T_REF}" "${GR00T_REPO_URL}" /workspace/Isaac-GR00T && \
      uv sync --project /workspace/Isaac-GR00T --python /workspace/.venv/bin/python && \
      uv pip install --python /workspace/.venv/bin/python -e /workspace/Isaac-GR00T; \
    fi

# Support PyTorch checkpoints (same behavior as upstream serve_policy image).
RUN /workspace/.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /workspace/src/openpi/models_pytorch/transformers_replace/* {}

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"
ENV OPENPI_DATA_HOME="/policy_assets"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
