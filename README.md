## EgoPHI: Estimating 3D Hand-Object Contact and Force from Egocentric Vision (ECCV 2026)

[Andela Ilic](), [Rachel Schuchert](), [Yijing Jiang](https://www.yijingjiang.com/), [Christian Holz](https://www.christianholz.net)<br/>

[Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland <br/>

___________

Our model **EgoPHI** is the first vision-based model that estimates 3D forces and contact during interaction with articulated rigid objects. Given a single egocentric monocular RGB image and the object geometry, EgoPHI registers hand meshes to refine the object pose in a shared camera coordinate frame, and predicts dense per-vertex contact and force distributions over all meshes for physically grounded interaction reasoning.
<div align="center">
<img src="figures/teaser.png" width="1200">
</div>


Abstract
----------
Understanding hand-object interaction from egocentric vision is essential for modeling how people physically engage with the surrounding world.
Yet reasoning about physically grounded interaction requires estimating the forces acting on hands and objects, beyond localizing contact.
We present EgoPHI, the first method that jointly estimates dense contact maps and 3D force distributions on hand and object meshes from a single monocular RGB image and object geometry.
To address the lack of scalable ground-truth force annotations, we introduce a physics-based simulation pipeline that augments existing hand-object datasets with dense per-vertex force supervision.
EgoPHI then learns dense 3D contact and force on interacting hand and articulated object meshes, extending vision-based force estimation beyond image-space or planar settings.
Our evaluation on in-distribution and out-of-distribution benchmarks shows that EgoPHI improves force estimation over existing approaches while generalizing to unseen datasets.
To evaluate sim-to-real transfer, we constructed two physical objects that capture dense object contact and force magnitude and used them to record a dataset of interactions from eight participants across diverse touch and grasp types.
Our results demonstrate that EgoPHI recovers meaningful 3D contact and force distributions in simulated, out-of-distribution, and real-world settings, advancing egocentric hand-object understanding from contact localization toward physically grounded interaction reasoning.


Method Overview
----------
EgoPHI's 3-stage pipeline: 
(1) visual & geometric feature extraction with cross-modal fusion, 
(2) object pose estimation, and 
(3) contact and force estimation. 
<p align="center">
<img src="figures/method.png" width="1200">
</p>

Code
----------

#### Repository layout

```
config.py               central path/hyperparameter configuration (see below)
general.py, angular.py  math utilities (rotation representations, etc.)
utils.py                shared helpers (MANO edge graphs, bbox extraction, losses, ...)
model.py                InteractionGNN model definition
dataset_ARCTIC.py       ARCTIC dataset/dataloader (training + evaluation, incl. HAMER hands)
dataset_H2O.py          H2O dataset/dataloader (evaluation only)
train.py                training entry point (ARCTIC)
evaluate_ARCTIC.py      evaluation entry point (ARCTIC val split)
evaluate_H2O.py         evaluation entry point (H2O, cross-dataset generalization)
compute_metrics.py      contact/force metrics from saved predictions
inspect_predictions.ipynb  visualize predictions for one chosen frame
arctic_preprocess.py    end-to-end ARCTIC preprocessing (resize, contacts, object pose, segmentation)
h2o_preprocess.py       end-to-end H2O preprocessing (resize, vertices, object pose, contacts, masks)
force_sim/              SOFA physics simulation that generates ARCTIC's force supervision
```

For force supervision, download the precomputed per-vertex force data from
HuggingFace and extract each into the matching path from `config.py`:
- [arctic_force_simulations.zip](https://huggingface.co/datasets/eth-siplab/EgoPHI/arctic_force_simulations.zip) -> `config.PROCESSED_FORCE_ROOT`
- [h2o_force_simulations.zip](https://huggingface.co/datasets/eth-siplab/EgoPHI/h2o_force_simulations.zip) -> `config.H2O_PROCESSED_FORCE_ROOT`

#### Dependencies

1. Create a new conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate egophi
   ```

2. Clone [HACO_RELEASE](https://github.com/dqj5182/HACO_RELEASE) next to this repo (or anywhere) --
   it provides the `lib.core.config` module `utils.py` imports for a couple of
   shared loss/config utilities. Point `HACO_RELEASE_ROOT` at it if it isn't at
   the default location (see Configuration below).
3. Download the [ARCTIC](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) dataset.
4. Download the [H2O](https://taeinkwon.com/projects/h2o/) dataset.
5. Download pretrained weights from [here]() and place `best_EgoPHI.pth` (and
   optionally `last_EgoPHI.pth`) under `checkpoints/`.

#### Configuration

All filesystem paths are centralized in `config.py` as module-level constants,
each overridable with an environment variable so the same code runs
unmodified on another machine. The ones you're most likely to need:

| Variable | Purpose | Default |
|---|---|---|
| `EGOPHI_ROOT` | repo root | this file's directory |
| `EGOPHI_DATA_ROOT` | preprocessed ARCTIC data root | `<DATA_ROOT>/arctic_data` |
| `EGOPHI_DATA_ROOT_H2O` | H2O data root | see `config.py` |
| `EGOPHI_H2O_CONTACTS_ROOT` | H2O contact-label root | see `config.py` |
| `EGOPHI_HAMER_VERTS_ROOT` | off-the-shelf HAMER hand estimates (ARCTIC eval) | see `config.py` |
| `EGOPHI_CHECKPOINT_DIR` | where checkpoints are read/written | `<repo>/checkpoints` |
| `HACO_RELEASE_ROOT` | path to the cloned HACO_RELEASE dependency | `<repo>/../HACO_RELEASE` |
| `ARCTIC_ROOT` | path to the upstream ARCTIC codebase (for segmentation rendering) | see `config.py` |

Read `config.py` for the complete list and the expected on-disk data layout.

#### Data preprocessing

Preprocessing produces everything the datasets/dataloaders read (resized
224px images, hand/object segmentation masks, object rotation/translation/
articulation, contact labels); force data comes separately, see above.

```bash
python arctic_preprocess.py     # runs all stages: resize, contacts, metadata, segmentation
python h2o_preprocess.py        # runs all stages: resize, vertices, transforms, contacts, segmentation
```

Both scripts accept `--stages` to run a subset (e.g. `--stages metadata`),
`--overwrite` to redo existing outputs, and root-path overrides -- run with
`--help` for the full list.

#### Training

```bash
python train.py
```

Trains `InteractionGNN` on the ARCTIC training split; the best/last
checkpoints are written to `config.CHECKPOINT_DIR`.

#### Evaluation

```bash
python evaluate_ARCTIC.py   # ARCTIC val split (s05)
python evaluate_H2O.py      # H2O, cross-dataset generalization
```

Both scripts read a checkpoint from `config.LEGACY_CHECKPOINT_PATH` if
present, else `config.BEST_CHECKPOINT_PATH`, and write per-frame predictions
under `evaluation_results/{arctic,h2o}/<sequence>/`. Useful overrides:
`EGOPHI_EVAL_DEVICE`, `EGOPHI_EVAL_CHECKPOINT`, `EGOPHI_EVAL_OUTPUT_DIR`,
`EGOPHI_EVAL_WORKERS`, and `EGOPHI_EVAL_MAX_SAMPLES` (stop after N samples,
useful for a quick smoke test).

Given those predictions and a matching folder of ground-truth `.pt` files,
`python compute_metrics.py --predictions-dir <dir> --gt-dir <dir>` prints
contact (precision/recall/F1/IoU) and force (masked MAE/RMSE, volumetric IoU)
metrics. `inspect_predictions.ipynb` visualizes a single chosen frame's
predictions instead of running the full evaluation.

Dataset
----------
This project trains on [ARCTIC](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)
and evaluates on both the ARCTIC held-out participant and [H2O](https://taeinkwon.com/projects/h2o/)
for cross-dataset generalization -- see Dependencies above for download links,
and Data preprocessing above for turning the raw downloads into the format
the dataloaders expect.

For sim-to-real evaluation, we also release [egophi_dataset.zip](https://huggingface.co/datasets/eth-siplab/EgoPHI/egophi_dataset.zip),
our real-world recordings on two physical objects with dense object-mesh
contact and force-magnitude ground truth.

Citation
----------
If you find our paper or codes useful, please cite our work:

    @article{
     }


License and Acknowledgement
----------
This project is released under the MIT license.
