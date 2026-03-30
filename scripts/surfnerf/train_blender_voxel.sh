#!/usr/bin/env bash
set -euo pipefail

# Example: no surface regularization + voxel hash grid.
DATA_DIR="${DATA_DIR:-/path/to/refnerf}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/surface_reg_release}"
SCENE="${SCENE:-lego}"

python -m train \
  --gin_configs=configs/ziprefnerf/blender.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/voxel_no_surface/${SCENE}'"