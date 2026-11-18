#!/usr/bin/env bash
# Serve ONE open-weight model on ONE GPU with vLLM (OpenAI-compatible API).
#   scripts/serve_llm.sh <hf-model-id> <gpu-index> <port> [extra vllm args...]
#   scripts/serve_llm.sh Qwen/Qwen3-8B 0 8101   (port 8000 is taken by a Jupyter server)
# Rule: a model never spans several GPUs. CUDA_VISIBLE_DEVICES pins it to one;
# tensor parallelism stays at 1.
set -euo pipefail
MODEL=${1:?model id}; GPU=${2:?gpu index}; PORT=${3:?port}; shift 3
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH; export PYTHONNOUSERSITE=1
export HF_HOME="$ROOT/data_cache/hf" HF_HUB_OFFLINE=1          # weights already downloaded
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm" XDG_CACHE_HOME="$ROOT/.cache" TRITON_CACHE_DIR="$ROOT/.cache/triton"
export HOME="$ROOT/.cache/home" FLASHINFER_WORKSPACE_BASE="$ROOT/.cache"   # libraries that write to ~ (home is read-only here)
mkdir -p "$HOME"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$ROOT/runs/llm_logs"
exec "$ROOT/.venv-vllm/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 2048 --max-num-seqs 8 \
  --served-model-name "$MODEL" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  "$@"
# The last two flags switch on tool calling: the server turns the model's
# <tool_call>{...}</tool_call> text into the reply's `tool_calls` field. `hermes` is the
# format the Qwen models write.
