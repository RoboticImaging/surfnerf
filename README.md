# Surf-NeRF: Code Release

This repository contains the JAX implementation for Surf-NeRF, including a minimal public training script surface for the core ablations used in the paper. This codebase builds heavily on the [ZipNeRF](https://github.com/jonbarron/camp_zipnerf) and [Ref-NeRF](https://github.com/google-research/multinerf) codebases, which provide the foundational NeRF framework, multi-scale rendering infrastructure, and reference view rendering techniques.

## Setup

```bash
# Clone the repo.
git clone https://github.com/nackjaylor/swiss_army_nerf_zip.git
cd swiss_army_nerf_zip

# Make and activate a conda environment.
conda create --name surfnerf python=3.11
conda activate surfnerf

# Install requirements.
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install pycolmap (this repo expects rmbrualla's pycolmap).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm unit tests pass.
./scripts/run_all_unit_tests.sh
```

You may also need a JAX build that supports your accelerator setup:
https://jax.readthedocs.io/en/latest/installation.html

## Datasets

Surf-NeRF datasets are available from the project page:
https://roboticimaging.org/Projects/SurfNeRF/

## Running

This release keeps a focused set of four Blender training variants:

- scripts/surface_reg/train_blender_voxel.sh
- scripts/surface_reg/train_blender_permuto.sh
- scripts/surface_reg/train_blender_surface_voxel.sh
- scripts/surface_reg/train_blender_surface_permuto.sh

Each script accepts optional environment variables:

- DATA_DIR (default: /path/to/refnerf)
- CHECKPOINT_DIR (default: /tmp/surface_reg_release)
- SCENE (default: lego)

Example:

```bash
DATA_DIR=/data/refnerf \
CHECKPOINT_DIR=/data/checkpoints \
SCENE=lego \
./scripts/surface_reg/train_blender_surface_permuto.sh
```

## Acknowledgements
The authors gratefully acknowledge our anonymous reviewers for their comments which improved our manuscript. This research was supported in part by funding from Ford Motor Company and the ARC Research Hub in Intelligent Robotic Systems for Real-Time Asset Management (IH210100030). Computational aspects of the work were supported in part through the NVIDIA Academic Grant Program and with Cloud TPUs from Google’s TPU Research Cloud (TRC). 

## Citation

If you use this codebase, please cite Surf-NeRF:

```bibtex
@inproceedings{naylor2026surf,
  title = {{Surf-NeRF}: Surface Regularised Neural Radiance Fields},
  author = {Naylor, Jack and Ila, Viorela and Dansereau, Donald G.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = jun,
  year = {2026},
}
```

This codebase also builds on ZipNeRF and Ref-NeRF. If you use this code, please also cite:

```bibtex
@article{barron2023zipnerf,
    title={Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields},
    author={Jonathan T. Barron and Ben Mildenhall and 
            Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
    journal={ICCV},
    year={2023}
}

@article{verbin2022refnerf,
    title={{Ref-NeRF}: Structured View-Dependent Appearance for
           Neural Radiance Fields},
    author={Dor Verbin and Peter Hedman and Ben Mildenhall and
            Todd Zickler and Jonathan T. Barron and Pratul P. Srinivasan},
    journal={CVPR},
    year={2022}
}
```
