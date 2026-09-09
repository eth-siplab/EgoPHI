"""
Central configuration for the EgoPHI training pipeline.

Every path used anywhere in the codebase is defined here, relative to two
roots:

    DATA_ROOT       -- where the (preprocessed) ARCTIC data lives
    CHECKPOINT_DIR   -- where trained model checkpoints get written

Both default to subfolders of PROJECT_ROOT but can be overridden with
environment variables, so the same code runs unmodified on another machine:

    EGOPHI_DATA_ROOT=/path/to/arctic_data \
    EGOPHI_CHECKPOINT_DIR=/path/to/checkpoints \
    python train.py

Expected DATA_ROOT layout (mirrors the original ARCTIC + preprocessing
output structure):

    arctic_data/
      data/
        images224/                                 (IMAGES_ROOT)
        body_models/mano/                          (MANO_ROOT)
        meta/object_vtemplates/                    (MESH_ROOT)
      render_out/                                  (SEGMENT_ROOT)
      outputs/
        manually_extracted_contacts/               (CONTACTS_ROOT)
        force_simulation/
          final_force_simulations_0.02_0.02/       (FORCE_ROOT)
          min_max_magnitude_per_folder/             (FORCE_MIN_MAX_MAGNITUDE_ROOT)
        processed_distance/                        (DISTANCES_ROOT)
        processed_verts/seqs/                      (PROCESSED_SEQS_ROOT)
        extracted_GT_MANO_params/                  (GT_MANO_ROOT)
        extracted_rot_trans_arti/                  (ROT_TRANS_ARTI_ROOT)
        HACO_contacts/                             (HACO_CONTACTS_ROOT)
        object_contact_stats/                      (OBJECT_CONTACT_STATS_DIR)
        object_means/                              (OBJECT_MEANS_DIR)
        force_means/                               (FORCE_MEANS_DIR)
      SOFA_force_log_magnitude_stats.json          (SOFA_FORCE_LOG_MAGNITUDE_STATS_JSON)
"""

import os

# --- Top-level roots --------------------------------------------------------
PROJECT_ROOT = os.environ.get("EGOPHI_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/pub/scratch/anilic/"
ARCTIC_DATA_ROOT = os.environ.get("EGOPHI_DATA_ROOT", os.path.join(DATA_ROOT, "arctic_data"))
CHECKPOINT_DIR = os.environ.get("EGOPHI_CHECKPOINT_DIR", os.path.join(PROJECT_ROOT, "checkpoints"))

# Third-party code this project depends on but doesn't vendor (clone separately)
HACO_RELEASE_ROOT = os.environ.get("HACO_RELEASE_ROOT", "/local/home/anilic/HACO_RELEASE")
ARCTIC_ROOT = os.environ.get("ARCTIC_ROOT", "/local/home/anilic/arctic")

# --- ARCTIC data (raw / provided) -------------------------------------------
IMAGES_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "images224")
MESH_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "meta", "object_vtemplates")
MANO_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "body_models", "mano")

# --- Preprocessing outputs consumed by the dataset --------------------------
SEGMENT_ROOT = os.path.join("/local/home/anilic/arctic/render_out")
CONTACTS_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "manually_extracted_contacts")
FORCE_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "force_simulation", "final_force_simulations_0.02_0.02")
FORCE_MIN_MAX_MAGNITUDE_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "force_simulation", "min_max_magnitude_per_folder")
DISTANCES_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "processed_distance")
PROCESSED_SEQS_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "processed_verts", "seqs")
GT_MANO_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "extracted_GT_MANO_params")
ROT_TRANS_ARTI_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "extracted_rot_trans_arti")
HACO_CONTACTS_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "HACO_contacts")

# --- Loss-normalization statistics ------------------------------------------
OBJECT_CONTACT_STATS_DIR = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "object_contact_stats")
OBJECT_MEANS_DIR = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "object_means")
FORCE_MEANS_DIR = os.path.join(ARCTIC_DATA_ROOT, "data", "outputs", "force_means")

# Global max force magnitudes used to normalize ARCTIC forces to [0, 1] --
# empirically measured over the ARCTIC force dataset, kept as a named
# constant instead of an inline magic number. Do not change these without
# re-checking what any existing trained model expects.
ARCTIC_FORCE_MAX_MAGNITUDE = {
    'left': 40786.97325154768,
    'right': 39529.099847565725,
    'object': 10540.6126048457,
}
SOFA_FORCE_LOG_MAGNITUDE_STATS_JSON = os.path.join(PROJECT_ROOT, "SOFA_force_log_magnitude_stats.json")

# --- Checkpoint filenames ----------------------------------------------------
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_EgoPHI.pth")
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last_EgoPHI.pth")
LEGACY_CHECKPOINT_PATH = os.environ.get(
  "EGOPHI_LEGACY_CHECKPOINT",
  "/local/home/anilic/arctic/checkpoints/video_joints_input_hand_distances_output/"
  "best_interactionGNN_ViT_with_obj_pose_est_AS_enhanced_26K.pth",
)

# --- HAMER (off-the-shelf hand-pose estimates), used for evaluation only ----
# ARCTIC-side HAMER MANO parameters, relative to ARCTIC_DATA_ROOT.
HAMER_MANO_ROOT = os.path.join(ARCTIC_DATA_ROOT, "data", "HAMER_MANO_params")
HAMER_VERTS_ROOT = os.environ.get(
  "EGOPHI_HAMER_VERTS_ROOT",
  "/local/home/anilic/arctic/data/HAMER_sipml3",
)

# --- H2O dataset (evaluation only) ------------------------------------------
# H2O lives on a separate data root from ARCTIC -- override with
# EGOPHI_DATA_ROOT_H2O if it's not next to arctic_data.
DATA_ROOT_H2O = os.environ.get(
    "EGOPHI_DATA_ROOT_H2O",
  "/pub/scratch/anilic/HACO_RELEASE/data/data/H2O/data",
)
H2O_IMAGES_ROOT = DATA_ROOT_H2O
H2O_CONTACTS_ROOT = os.environ.get(
  "EGOPHI_H2O_CONTACTS_ROOT",
  "/local/home/anilic/EgoPHI_old/HACO_RELEASE/outputs/contacts",
)
H2O_FORCE_ROOT = os.path.join(DATA_ROOT_H2O, "forces_rigid_0.04_0.04")
H2O_ROT_TRANS_SCALE_ROOT = os.path.join(DATA_ROOT_H2O, "extracted_rot_trans_scale")
H2O_HAMER_ROOT = os.path.join(DATA_ROOT_H2O, "HAMER_sipml3", "hamer_out")

# Global max force magnitudes used to normalize H2O forces to [0, 1] --
# empirically measured over the H2O force dataset, kept as named constants
# instead of inline magic numbers. Do not change these without re-checking
# what any existing trained model expects.
H2O_FORCE_MAX_MAGNITUDE = {
    'left': 139097.6954896031,
    'right': 115636.71941756597,
    'object': 8427.382710683385,
}