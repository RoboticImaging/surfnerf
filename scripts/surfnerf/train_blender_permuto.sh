#!/usr/bin/env bash
set -euo pipefail

# Example: no surface regularization + permutohedral hash grid.
DATA_DIR="${DATA_DIR:-/path/to/refnerf}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/surface_reg_release}"
SCENE="${SCENE:-lego}"

python -m train \
  --gin_configs=configs/ziprefnerf/blender.gin \
  --gin_bindings="Config.hash_grid_type = 'permuto'" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/permuto_no_surface/${SCENE}'"