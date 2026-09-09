"""
Visualize predicted per-vertex contact and force on a single H2O sequence.

Unlike `evaluate_H2O.py`, this does NOT construct the full
`H2OSequenceDataset` -- that class's `__init__` scans every force file for
the chosen subject(s) (needed to reproduce the training/eval split exactly),
which is unnecessary work here: this script only ever needs the model's
*inputs* for one sequence, never any ground truth. Instead it loads the same
per-frame files `H2OSequenceDataset` would, directly, for just the chosen
(participant, object) pair -- reusing the dataset class's own mesh/camera/
HAMER-alignment loading methods (via a "bare" instance that skips
`__init__`) so the loading math is identical, without the expensive scan.

H2O has no ground-truth hand mesh, so (as in `evaluate_H2O.py`) the model is
always fed HAMER-estimated hand vertices.

`prepare_sequence` / `run_frame` below do the actual data loading and
inference for one sequence / one frame and are reused as-is by
`inspect_predictions.ipynb` to render a single chosen frame -- see that
notebook for an interactive, single-frame version of this script.
"""

import glob
import os
from pathlib import Path

import torch
import trimesh
from manopth.manolayer import ManoLayer
from torchvision.io import read_image
from tqdm import tqdm

import config
from dataset_H2O import H2OSequenceDataset
from model import InteractionGNN
from utils import extract_2d_bboxes, prepare_mano_edges, render_prediction_figure, to_device

H_ORIG, W_ORIG = 720, 1280


def first_tracked_frame(hamer_root, object_key):
    """Index of the first frame position with a real HAMER estimate for both hands (see visualize_arctic.py)."""
    hamer_dir = os.path.join(hamer_root, object_key)
    left_ids = {int(Path(p).stem.split("_")[-1]) for p in glob.glob(os.path.join(hamer_dir, "vert_0.0_*.npy"))}
    right_ids = {int(Path(p).stem.split("_")[-1]) for p in glob.glob(os.path.join(hamer_dir, "vert_1.0_*.npy"))}
    both = left_ids & right_ids
    return min(both) if both else 0


def bare_dataset():
    """An `H2OSequenceDataset` with `__init__` skipped -- see visualize_arctic.py's `bare_dataset` for why."""
    dataset = object.__new__(H2OSequenceDataset)
    dataset.images_root = config.H2O_IMAGES_ROOT
    dataset.rot_trans_scale_root = config.H2O_ROT_TRANS_SCALE_ROOT
    dataset.hamer_root = config.H2O_HAMER_ROOT
    dataset.cameras = ['0']
    return dataset


def load_model(device, checkpoint_path=None):
    """Builds InteractionGNN and loads the eval checkpoint (same defaults as `evaluate_H2O.py`)."""
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


def prepare_sequence(participant, object_key, start_frame=None):
    """
    Loads everything about one (participant, object_key) sequence that
    doesn't change frame to frame. `object_key` is the per-subject relative
    path (e.g. "h1/0/cam4"); `full_key` (participant prepended) is what
    dataset_H2O.py's own loader methods actually key by -- see
    `_index_sequences`' `object_key` construction there.
    """
    full_key = f"{participant}/{object_key}"
    dataset = bare_dataset()
    coords, edges = dataset._load_articulated_mesh(full_key)
    # _load_articulated_mesh returns edges as a raw numpy [E, 2] array (from
    # trimesh's edges_unique) -- evaluate_H2O.py transposes this to the [2, E]
    # edge_index format the model expects via `.permute(1, 0)` after
    # DataLoader's default collation turns it into a tensor; do the same here.
    edges = torch.as_tensor(edges, dtype=torch.long).T
    obj_faces = trimesh.load(
        os.path.join(config.H2O_IMAGES_ROOT, full_key, 'object_frame0_new.obj'), process=False
    ).faces
    K_ego = dataset._load_cam_int(full_key)

    if start_frame is None:
        start_frame = first_tracked_frame(config.H2O_HAMER_ROOT, full_key)

    image_paths = sorted(glob.glob(os.path.join(config.H2O_IMAGES_ROOT, participant, object_key, 'rgb224', '*.png')))
    mask_dir = os.path.join(config.H2O_IMAGES_ROOT, participant, object_key, 'object_mask224')
    mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in image_paths]
    if not image_paths or not all(os.path.isfile(p) for p in mask_paths):
        raise FileNotFoundError(
            f"No frames (or no matching masks) found for participant={participant!r} "
            f"object={object_key!r}. Check the names and that preprocessing has been run."
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
        'dataset': dataset, 'participant': participant, 'object_key': object_key, 'full_key': full_key,
        'coords': coords, 'edges': edges, 'obj_faces': obj_faces, 'K_ego': K_ego,
        'image_paths': image_paths, 'mask_paths': mask_paths, 'start_frame': start_frame,
        'hand_edges_left': hand_edges_left, 'hand_edges_right': hand_edges_right,
        'faces_left': faces_left, 'faces_right': faces_right, 'hand_faces': hand_faces,
    }


def run_frame(seq, model, device, frame_position):
    """
    Runs the model on one frame of a sequence prepared by `prepare_sequence`.
    `frame_position` indexes directly into `seq['image_paths']`/`seq['mask_paths']`
    and into the per-object rot/trans/scale arrays (same position convention
    throughout dataset_H2O.py).

    Returns a dict of numpy arrays ready for rendering -- see
    visualize_arctic.py's `run_frame` for the field list, which this mirrors.
    """
    dataset = seq['dataset']

    (world_verts, left_verts, right_verts, scale,
     hamer_left, hamer_right, hamer_left_aligned, hamer_right_aligned,
     R, T) = dataset._load_obj_R_T_arti(seq['full_key'], frame_position, 1)

    image = read_image(seq['image_paths'][frame_position]).float().unsqueeze(0) / 255.0
    # object_mask224 files are single-channel, so this is already [1, H, W]
    # -- matching the [num_frames=1, H, W] extract_2d_bboxes expects.
    object_mask = read_image(seq['mask_paths'][frame_position]).float() / 255.0
    object_bbox = extract_2d_bboxes(object_mask)

    vert_left = hamer_left_aligned.reshape(1, 778, 3)
    vert_right = hamer_right_aligned.reshape(1, 778, 3)

    verts_l_center = torch.mean(vert_left, dim=1)
    verts_r_center = torch.mean(vert_right, dim=1)

    (image, vert_left, vert_right, edges_dev, obj_template_vert, K_ego_dev) = to_device(
        device, image, vert_left, vert_right, seq['edges'], seq['coords'].unsqueeze(0), seq['K_ego'].unsqueeze(0),
    )
    verts_l_center, verts_r_center, object_bbox, scale = to_device(
        device, verts_l_center, verts_r_center, object_bbox, scale,
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
            obj_template_vert, scale, object_bbox, K_ego_dev,
            seq['hand_edges_left'], seq['hand_edges_right'], edges_dev,
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
        'participant': seq['participant'], 'object_key': seq['object_key'],
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
        os.path.join(config.PROJECT_ROOT, "visualizations_h2o"),
    )
    num_frames = int(os.environ.get("EGOPHI_VIS_NUM_FRAMES", "100"))
    participant = os.environ.get("EGOPHI_VIS_PARTICIPANT", "subject4_ego")
    object_key = os.environ.get("EGOPHI_VIS_OBJECT", "h1/0/cam4")
    env_start_frame = os.environ.get("EGOPHI_VIS_START_FRAME")

    model = load_model(device)
    print('MODEL LOADED')

    seq = prepare_sequence(participant, object_key, start_frame=int(env_start_frame) if env_start_frame else None)
    num_frames = min(num_frames, len(seq['image_paths']) - seq['start_frame'])
    if num_frames <= 0:
        raise FileNotFoundError(
            f"No frames left for participant={participant!r} object={object_key!r} "
            f"starting at frame {seq['start_frame']}."
        )
    print(f"Starting at frame position {seq['start_frame']} (first with tracked hands)")

    filename = seq['full_key'].replace('/', '_')
    for frame_idx in tqdm(range(num_frames), desc="Visualizing"):
        result = run_frame(seq, model, device, seq['start_frame'] + frame_idx)

        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        render_prediction_figure(
            os.path.join(save_dir, f"frame_{result['frame_id']}.png"),
            result['rgb_image'],
            result['hand_vertices'], result['hand_faces'], result['hand_contact'],
            result['hand_force_mag'], result['hand_force_dir'],
            result['object_vertices'], result['object_faces'], result['object_contact'],
            result['object_force_mag'], result['object_force_dir'],
            title=f"{participant}/{object_key} / {result['frame_id']}",
        )


if __name__ == '__main__':
    main()
