#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Mamba + Transformer Hybrid-Model Environment Setup
# ============================================================
# This script does NOT create or activate a conda environment.
# You are responsible for that — create and activate it yourself
# first, e.g.:
#
#   conda create -n mamba-env python=3.10
#   conda activate mamba-env
#   ./setup_mamba_env.sh
#
# Recommended Python: 3.10. It has full pre-built wheel coverage
# for torch 2.4 / mamba-ssm 2.2.x / causal-conv1d 1.4.0, so you
# avoid triggering from-source builds or ABI mismatches. 3.11
# mostly works but wheel coverage is spottier; 3.9 and below miss
# recent transformers features.
#
# Designed to be idempotent and reproducible:
#   - all critical packages are version-pinned
#   - CUDA compiler + toolkit are installed INSIDE the active env
#     (no dependency on a system-wide CUDA path like ~/cuda-12.4)
#   - safe to re-run inside a brand new conda env months from now
# ============================================================

# ---- Pinned, known-compatible versions ----
# Bump these deliberately (and re-test) rather than letting pip
# silently grab whatever is newest at install time.
TORCH_VERSION="2.4.0"
CUDA_TAG="cu121"                 # must match CUDATOOLKIT_VERSION below
CUDATOOLKIT_VERSION="12.1"
MAMBA_SSM_VERSION="2.2.4"
CAUSAL_CONV1D_VERSION="1.4.0"
TRANSFORMERS_VERSION="4.44.2"
ACCELERATE_VERSION="0.33.0"

log()  { echo -e "\n\033[1;36m▶ $1\033[0m"; }
err()  { echo -e "\033[1;31m✖ $1\033[0m" >&2; }
trap 'err "Failed at line $LINENO. See output above for details."' ERR

# ------------------------------------------------------------
# 0. Sanity checks
# ------------------------------------------------------------
if ! command -v conda &> /dev/null; then
  err "conda not found in PATH. Install Miniconda/Anaconda first."
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-base}" == "base" ]]; then
  err "No non-base conda environment is active."
  err "Create and activate one first, e.g.:"
  err "  conda create -n mamba-env python=3.10 && conda activate mamba-env"
  exit 1
fi

log "Active env: ${CONDA_DEFAULT_ENV} ($(which python))"
python -c "import sys; print('Python:', sys.version.split()[0])"

# ------------------------------------------------------------
# 2. Self-contained CUDA toolchain (inside the env, not ~/cuda-x)
# ------------------------------------------------------------
# Main fix vs. the old script: a hardcoded path like ~/cuda-12.4
# breaks the moment that folder is missing, renamed, or absent on
# a new machine. Installing the compiler + CUDA toolkit as conda
# packages keeps everything inside the env, so it survives
# indefinitely across new environments/machines.
log "Installing compilers + CUDA toolkit inside the env..."
conda install -y -c conda-forge -c nvidia \
  compilers \
  "cuda-toolkit=${CUDATOOLKIT_VERSION}" \
  "cuda-nvcc=${CUDATOOLKIT_VERSION}"

export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CC:   ${CC}"
echo "nvcc: $(which nvcc)"
nvcc -V

# Auto-detect GPU compute capability; fall back to a broad list
# covering Ampere/Ada/Hopper if detection fails (e.g. no GPU
# visible at build time).
DETECTED_ARCH=""
if command -v nvidia-smi &> /dev/null; then
  DETECTED_ARCH=$(python - <<'PYEOF'
import subprocess
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    ).decode().strip().splitlines()
    caps = sorted(set(c.strip() for c in out if c.strip()))
    print(";".join(caps))
except Exception:
    print("")
PYEOF
)
fi
export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH:-8.0;8.6;8.9;9.0}"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# ------------------------------------------------------------
# 3. PyTorch (pinned; must be installed before mamba/causal-conv1d)
# ------------------------------------------------------------
log "Installing PyTorch ${TORCH_VERSION}+${CUDA_TAG}..."
pip install \
  "torch==${TORCH_VERSION}" torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python - <<'PYEOF'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
PYEOF

# ------------------------------------------------------------
# 4. Build dependencies
# ------------------------------------------------------------
log "Installing build dependencies..."
pip install ninja einops packaging setuptools wheel psutil

# ------------------------------------------------------------
# 5. Clean any previous conflicting installs (safe if absent)
# ------------------------------------------------------------
log "Cleaning any previous mamba/causal-conv1d/triton installs..."
pip uninstall -y mamba-ssm causal-conv1d triton || true
pip cache purge || true

# ------------------------------------------------------------
# 6. causal-conv1d (mamba-ssm's fused CUDA conv kernel dependency)
# ------------------------------------------------------------
log "Installing causal-conv1d ${CAUSAL_CONV1D_VERSION}..."
pip install "causal-conv1d==${CAUSAL_CONV1D_VERSION}" \
  --no-build-isolation --no-cache-dir

# ------------------------------------------------------------
# 7. mamba-ssm
# ------------------------------------------------------------
log "Installing mamba-ssm ${MAMBA_SSM_VERSION}..."
pip install "mamba-ssm==${MAMBA_SSM_VERSION}" \
  --no-build-isolation --no-cache-dir

# ------------------------------------------------------------
# 8. Hybrid transformer + mamba stack
# ------------------------------------------------------------
log "Installing transformers stack for hybrid transformer/mamba models..."
pip install \
  "transformers==${TRANSFORMERS_VERSION}" \
  "accelerate==${ACCELERATE_VERSION}" \
  tokenizers safetensors datasets huggingface_hub sentencepiece

# ------------------------------------------------------------
# 9. Verification (imports + a real hybrid forward pass)
# ------------------------------------------------------------
log "Verifying installation..."
python - <<'PYEOF'
import torch
import mamba_ssm
import causal_conv1d
import transformers
import torch.nn as nn
from mamba_ssm import Mamba

print("Torch:         ", torch.__version__)
print("CUDA avail:    ", torch.cuda.is_available())
print("transformers:  ", transformers.__version__)

try:
    import mamba_ssm.ops.selective_scan_interface  # noqa: F401
    print("Mamba CUDA kernels: OK")
except Exception as e:
    print("Mamba CUDA kernels NOT compiled:", e)

# Hybrid sanity check: a Mamba block and a Transformer encoder
# layer both processing the same (batch, seq, d_model) tensor,
# the way a hybrid block-interleaved model would use them.
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(2, 16, 64, device=device)

mamba_block = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
transformer_block = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, batch_first=True
).to(device)

y_mamba = mamba_block(x)
y_attn = transformer_block(x)
print("Hybrid sanity check — Mamba out:", tuple(y_mamba.shape),
      "| Transformer out:", tuple(y_attn.shape))
PYEOF

log "Done. Environment '${ENV_NAME}' is ready for hybrid Mamba/Transformer work."
echo "Activate it anytime with: conda activate ${ENV_NAME}"