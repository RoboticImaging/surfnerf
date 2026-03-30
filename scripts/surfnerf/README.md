# Surface Reg Release Scripts

This folder contains the minimal training script set used for public release.

## Variants

1. `train_blender_voxel.sh`: surface regularization off, voxel grid.
2. `train_blender_permuto.sh`: surface regularization off, permutohedral grid.
3. `train_blender_surface_voxel.sh`: surface regularization on, voxel grid.
4. `train_blender_surface_permuto.sh`: surface regularization on, permutohedral grid.

## Usage

```bash
DATA_DIR=/path/to/refnerf \
CHECKPOINT_DIR=/path/to/checkpoints \
SCENE=lego \
./scripts/surface_reg/train_blender_surface_permuto.sh
```

Defaults used by all scripts:

- `DATA_DIR=/path/to/refnerf`
- `CHECKPOINT_DIR=/tmp/surface_reg_release`
- `SCENE=lego`
