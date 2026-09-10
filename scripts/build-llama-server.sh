#!/usr/bin/env bash
# Build llama-server (CUDA) for the Needle agent harness.
# Usage: scripts/build-llama-server.sh
# Env overrides: CUDAToolkit_ROOT, CMAKE_BIN, LLAMA_SRC_REF
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${CUDAToolkit_ROOT:=/usr/local/cuda}"
: "${CMAKE_BIN:=cmake}"
: "${LLAMA_SRC_REF:=}"

if [ ! -f "$REPO/third_party/llama.cpp/CMakeLists.txt" ]; then
  echo "Cloning llama.cpp into third_party/llama.cpp ..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$REPO/third_party/llama.cpp"
fi
if [ -n "$LLAMA_SRC_REF" ]; then
  git -C "$REPO/third_party/llama.cpp" fetch --depth 1 origin "$LLAMA_SRC_REF"
  git -C "$REPO/third_party/llama.cpp" checkout "$LLAMA_SRC_REF"
fi

"$CMAKE_BIN" -S "$REPO/third_party/llama.cpp" -B "$REPO/third_party/llama.cpp/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT" \
  -DCMAKE_CUDA_COMPILER="$CUDAToolkit_ROOT/bin/nvcc"

"$CMAKE_BIN" --build "$REPO/third_party/llama.cpp/build" --target llama-server

echo "Built: $REPO/third_party/llama.cpp/build/bin/llama-server"
