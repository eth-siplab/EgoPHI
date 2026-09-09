"""
Visualize predicted per-vertex contact and force on a single ARCTIC sequence.

Unlike `evaluate_ARCTIC.py`, this does NOT construct the full
`ArcticSequenceDataset` -- that class's `__init__` scans force files across
every participant in the dataset (needed to reproduce the legacy training
split exactly, see dataset_ARCTIC.py's comments on `self.to_skip`), which
takes a long time and isn't needed here: this script only ever needs the
model's *inputs* for one sequence, never any ground truth. Instead it loads
the same per-frame files `ArcticSequenceDataset` would, directly, for just
the chosen (participant, object) pair -- reusing the dataset class's own
mesh/camera/HAMER-alignment loading methods (via a "bare" instance that skips
`__init__`) so the loading math is identical, without the expensive scan.

For each of the first `EGOPHI_VIS_NUM_FRAMES` frames of
`EGOPHI_VIS_PARTICIPANT` / `EGOPHI_VIS_OBJECT`, saves one figure with the
input frame, the MANO hand meshes colored by predicted contact/force, and the
object mesh colored by predicted contact/force.

`prepare_sequence` / `run_frame` below do the actual data loading and
inference for one sequence / one frame and are reused as-is by
`inspect_predictions.ipynb` to render a single chosen frame -- see that
notebook for a interactive, single-frame version of this script.
"""

import glob
import os
from pathlib import Path

import torch
import trimesh
from manopth.manolayer import ManoLayer
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import config
from dataset_ARCTIC import ArcticSequenceDataset
from model import InteractionGNN, apply_articulation_batch
from utils import extract_2d_bboxes, prepare_mano_edges, render_prediction_figure, to_device

USE_HAMER_HANDS = True
H_ORIG, W_ORIG = 2000, 2800


def first_tracked_frame(hamer_verts_root, object_name):
    """
    Index of the first frame position with a real HAMER hand-vertex estimate
    for both hands. Early frames of a grab sequence often have no usable hand
    detection yet (before the hand enters frame / tracking locks on) --
    `_load_obj_R_T_arti` silently falls back to an all-zero hand mesh for
    those (matching real eval behavior), which is correct but not a useful
    default starting point for a visualization. Falls back to 0 if nothing
    is found (e.g. HAMER wasn't run for this sequence at all).
    """
    hamer_dir = os.path.join(hamer_verts_root, object_name, "0")
    left_ids = {int(Path(p).stem.split("_")[-1]) for p in glob.glob(os.path.join(hamer_dir, "vert_0.0_*.npy"))}
    right_ids = {int(Path(p).stem.split("_")[-1]) for p in glob.glob(os.path.join(hamer_dir, "vert_1.0_*.npy"))}
    both = left_ids & right_ids
    return min(both) if both else 0


def bare_dataset():
    """
    An `ArcticSequenceDataset` with `__init__` skipped, carrying only the
    handful of attributes `_load_articulated_mesh`/`_load_cam_int`/
    `_load_obj_R_T_arti` actually read. `to_skip=[]` (no excluded frames) is
    fine here: it only ever excludes ~25 known-bad frames dataset-wide (out
    of tens of thousands), so for a chosen good sequence it's a no-op -- and
    unlike training/`evaluate_ARCTIC.py`, this script never touches the
    ground-truth force data that list was built from in the first place.
    """
    dataset = object.__new__(ArcticSequenceDataset)
    dataset.mesh_root = config.MESH_ROOT
    dataset.subj_path = config.ROT_TRANS_ARTI_ROOT
    dataset.hamer_verts_root = config.HAMER_VERTS_ROOT
    dataset.to_skip = []
    return dataset


def load_model(device, checkpoint_path=None):
    """Builds InteractionGNN and loads the eval checkpoint (same defaults as `evaluate_ARCTIC.py`)."""
    model = InteractionGNN(d_model=512, n_head=4, num_pose_blocks=4, num_force_blocks=4,
                            backbone_feature_dim=768, num_pose_iterations=3)
    model = model.to(device)
    checkpoint_path = checkpoint_path or os.environ.get(
        "EGOPHI_EVAL_CHECKPOINT",
        config.LEGACY_CHECKPOINT_PATH if os.path.isfile(config.LEGACY_CHECKPOINT_PATH)
        else config.BEST_CHECKPOINT_PATH,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


def prepare_sequence(participant, object_name, start_frame=None):
    """
    Loads everything about one (participant, object) sequence that doesn't
    change frame to frame: the object template mesh/edges, camera intrinsics,
    hand topology, and the full (unfiltered) list of image/mask paths.
    `start_frame` defaults to `first_tracked_frame` (see above) when omitted.
    """
    dataset = bare_dataset()
    coords, _faces, edge_index, _edge_weight, part_ids, pivot, _normals = dataset._load_articulated_mesh(object_name)
    obj_faces = trimesh.load(
        os.path.join(config.MESH_ROOT, object_name.split('_')[0], "mesh.obj"), process=False
    ).faces
    K_ego = dataset._load_cam_int(participant)

    if start_frame is None:
        start_frame = first_tracked_frame(config.HAMER_VERTS_ROOT, object_name)

    image_paths = sorted(glob.glob(os.path.join(config.IMAGES_ROOT, participant, object_name, '0', '*.jpg')))
    mask_paths = sorted(glob.glob(
        os.path.join(config.SEGMENT_ROOT, f"{participant}_{object_name}_0", "images", "mask_object", "*.png")
    ))
    if not image_paths or not mask_paths:
        raise FileNotFoundError(
            f"No frames found for participant={participant!r} object={object_name!r}. "
            "Check the names and that preprocessing has been run."
        )

    hand_edges_left, hand_edges_right, _, _ = prepare_mano_edges()
    faces_left = ManoLayer(side='left', use_pca=False, flat_hand_mean=False, ncomps=45,
                            mano_root=config.MANO_ROOT).th_faces.numpy()
    faces_right = ManoLayer(side='right', use_pca=False, flat_hand_mean=False, ncomps=45,
                             mano_root=config.MANO_ROOT).th_faces.numpy()
    hand_faces = torch.cat([
        torch.as_tensor(faces_left), torch.as_tensor(faces_right) + faces_left.max() + 1,
    ]).numpy()

    return {
        'dataset': dataset, 'participant': participant, 'object_name': object_name,
        'coords': coords, 'edge_index': edge_index, 'part_ids': part_ids, 'pivot': pivot,
        'obj_faces': obj_faces, 'K_ego': K_ego,
        'image_paths': image_paths, 'mask_paths': mask_paths, 'start_frame': start_frame,
        'hand_edges_left': hand_edges_left, 'hand_edges_right': hand_edges_right,
        'faces_left': faces_left, 'faces_right': faces_right, 'hand_faces': hand_faces,
    }


def run_frame(seq, model, device, frame_position, use_hamer_hands=USE_HAMER_HANDS):
    """
    Runs the model on one frame of a sequence prepared by `prepare_sequence`.
    `frame_position` indexes directly into `seq['image_paths']`/`seq['mask_paths']`
    and into the per-object rot/trans/arti arrays (same convention throughout
    dataset_ARCTIC.py -- see `first_tracked_frame`'s docstring for why this
    isn't simply the frame's number in its filename).

    Returns a dict of numpy arrays ready for rendering: hand/object vertices,
    faces, predicted contact, predicted force magnitude, predicted force unit
    direction, the input RGB frame, and the frame's id (image filename stem).
    """
    dataset = seq['dataset']
    participant, object_name = seq['participant'], seq['object_name']

    (R, T, s, arti, world_verts, left_verts, right_verts, world2ego, bbox3d,
     hamer_left, hamer_right) = dataset._load_obj_R_T_arti(participant, object_name, frame_position, 1)

    image = read_image(seq['image_paths'][frame_position]).float().unsqueeze(0) / 255.0
    object_mask = to_tensor(Image.open(seq['mask_paths'][frame_position]))[0].unsqueeze(0)
    object_bbox = extract_2d_bboxes(object_mask)

    # _load_obj_R_T_arti's hamer_left/right come back as [778, 3] when a real
    # HAMER file was loaded (the per-frame centering step collapses the
    # length=1 slice axis) but as [1, 778, 3] in its all-zero fallback (no
    # file for this frame) -- .reshape (same trick evaluate_H2O.py uses)
    # works for either shape since the element count (778*3) is the same.
    if use_hamer_hands:
        vert_left = hamer_left.reshape(1, 778, 3)
        vert_right = hamer_right.reshape(1, 778, 3)
    else:
        vert_left = left_verts
        vert_right = right_verts

    verts_l_center = torch.mean(vert_left, dim=1)
    verts_r_center = torch.mean(vert_right, dim=1)

    # pivot.pt is [3] on disk, but _load_articulated_mesh's centering step
    # (pivot - center) broadcasts it against center's [1, 3] shape, so the
    # pivot this function returns is already [1, 3] -- no unsqueeze.
    verts_articulated = apply_articulation_batch(
        vertices=seq['coords'].unsqueeze(0), part_ids=seq['part_ids'].unsqueeze(0), part_id_to_move=1,
        radians=arti, pivot=seq['pivot'], axis=None)

    (image, verts_l_center, verts_r_center, vert_left, vert_right,
     object_bbox, obj_template_vert, verts_articulated, edges,
     s, K_ego_dev) = to_device(
        device, image, verts_l_center, verts_r_center, vert_left, vert_right,
        object_bbox, seq['coords'].unsqueeze(0), verts_articulated, seq['edge_index'],
        s, seq['K_ego'].unsqueeze(0),
    )

    with torch.no_grad():
        (pred_contact_left, pred_contact_right, pred_contact_obj,
         pred_left_force_magnitude, pred_right_force_magnitude,
         pred_obj_force_magnitude,
         pred_l_force_vector, pred_r_force_vector, pred_o_force_vector,
         all_pred_trans, all_pred_rot,
         final_pred_trans, final_pred_rot,
         verts_o_pred) = model(
            image, vert_left, vert_right, verts_l_center, verts_r_center,
            verts_articulated, s, object_bbox, K_ego_dev,
            seq['hand_edges_left'], seq['hand_edges_right'], edges,
            H_ORIG, W_ORIG,
        )

    hand_vertices = torch.cat([vert_left[0], vert_right[0]], dim=0).cpu().numpy()
    hand_contact = torch.sigmoid(torch.cat([pred_contact_left[0], pred_contact_right[0]])).squeeze(-1).cpu().numpy()
    hand_force_mag = torch.sigmoid(
        torch.cat([pred_left_force_magnitude[0], pred_right_force_magnitude[0]])
    ).squeeze(-1).cpu().numpy()
    hand_force_vec = torch.cat([pred_l_force_vector[0], pred_r_force_vector[0]])
    hand_force_dir = (hand_force_vec / (hand_force_vec.norm(dim=-1, keepdim=True) + 1e-8)).cpu().numpy()

    object_vertices = verts_o_pred[0].detach().cpu().numpy()
    object_contact = torch.sigmoid(pred_contact_obj[0]).squeeze(-1).cpu().numpy()
    object_force_mag = torch.sigmoid(pred_obj_force_magnitude[0]).squeeze(-1).cpu().numpy()
    object_force_dir = (pred_o_force_vector[0] / (pred_o_force_vector[0].norm(dim=-1, keepdim=True) + 1e-8)).cpu().numpy()

    return {
        'participant': participant, 'object_name': object_name,
        'frame_id': Path(seq['image_paths'][frame_position]).stem,
        'rgb_image': image[0].permute(1, 2, 0).cpu().numpy(),
        'left_vertices': vert_left[0].cpu().numpy(), 'right_vertices': vert_right[0].cpu().numpy(),
        'faces_left': seq['faces_left'], 'faces_right': seq['faces_right'],
        'hand_vertices': hand_vertices, 'hand_faces': seq['hand_faces'],
        'hand_contact': hand_contact, 'hand_force_mag': hand_force_mag, 'hand_force_dir': hand_force_dir,
        'left_contact': torch.sigmoid(pred_contact_left[0]).squeeze(-1).cpu().numpy(),
        'right_contact': torch.sigmoid(pred_contact_right[0]).squeeze(-1).cpu().numpy(),
        'left_force_mag': torch.sigmoid(pred_left_force_magnitude[0]).squeeze(-1).cpu().numpy(),
        'right_force_mag': torch.sigmoid(pred_right_force_magnitude[0]).squeeze(-1).cpu().numpy(),
        'object_vertices': object_vertices, 'object_faces': seq['obj_faces'],
        'object_contact': object_contact, 'object_force_mag': object_force_mag, 'object_force_dir': object_force_dir,
    }


def main():
    device = torch.device(os.environ.get("EGOPHI_EVAL_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = os.environ.get(
        "EGOPHI_VIS_OUTPUT_DIR",
        os.path.join(config.PROJECT_ROOT, "visualizations_arctic"),
    )
    num_frames = int(os.environ.get("EGOPHI_VIS_NUM_FRAMES", "100"))
    participant = os.environ.get("EGOPHI_VIS_PARTICIPANT", "s05")
    object_name = os.environ.get("EGOPHI_VIS_OBJECT", "box_grab_01")
    env_start_frame = os.environ.get("EGOPHI_VIS_START_FRAME")

    model = load_model(device)
    print('MODEL LOADED')

    seq = prepare_sequence(participant, object_name, start_frame=int(env_start_frame) if env_start_frame else None)
    num_frames = min(num_frames, len(seq['image_paths']) - seq['start_frame'], len(seq['mask_paths']) - seq['start_frame'])
    if num_frames <= 0:
        raise FileNotFoundError(
            f"No frames left for participant={participant!r} object={object_name!r} "
            f"starting at frame {seq['start_frame']}."
        )
    print(f"Starting at frame position {seq['start_frame']} (first with tracked hands)")
    print(f'VISUALIZING {participant}/{object_name}, {num_frames} frames')

    for frame_idx in tqdm(range(num_frames), desc="Visualizing"):
        result = run_frame(seq, model, device, seq['start_frame'] + frame_idx)

        save_dir = os.path.join(output_dir, object_name)
        os.makedirs(save_dir, exist_ok=True)
        render_prediction_figure(
            os.path.join(save_dir, f"frame_{result['frame_id']}.png"),
            result['rgb_image'],
            result['hand_vertices'], result['hand_faces'], result['hand_contact'],
            result['hand_force_mag'], result['hand_force_dir'],
            result['object_vertices'], result['object_faces'], result['object_contact'],
            result['object_force_mag'], result['object_force_dir'],
            title=f"{participant}/{object_name} / {result['frame_id']}",
        )


if __name__ == '__main__':
    main()
