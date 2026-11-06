#!/usr/bin/env bash
# Compile every CUDA extension PriDE needs.  Run once from the repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1) splatting + multi-view sampling (installed as an editable plugin package)
cd "${ROOT}/projects"
python setup.py develop

# 2) backtrace packages -- built in place, because the heads import `_C` from
#    the source tree (projects/mmdet3d_plugin/ops/<pkg>/...), not site-packages.
for pkg in tile_localagg_prob_sq_bt tile_localagg_prob_sq_bt2; do
    cd "${ROOT}/projects/mmdet3d_plugin/ops/${pkg}"
    rm -rf build
    python setup.py build_ext --inplace
done

echo "All CUDA extensions built."
