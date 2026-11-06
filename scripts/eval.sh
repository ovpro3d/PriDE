#!/usr/bin/env bash
#
# Evaluate a trained PriDE checkpoint on the official nuScenes val split
# (150 scenes / 6,019 samples): mIoU, binary IoU and RayIoU.
#
#   bash scripts/eval.sh l work_dirs/pride-l_v11/train48e/latest.pth
#
set -euo pipefail
SIZE="${1:-l}"
CKPT="${2:?usage: bash scripts/eval.sh <l|m> <checkpoint.pth> [num_gpus]}"
GPUS="${3:-8}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CFG="projects/configs/superocc/pride-${SIZE}_r50_704_seq_nui_48e_v11.py"
bash tools/dist_test.sh "${CFG}" "${CKPT}" "${GPUS}" --eval map
