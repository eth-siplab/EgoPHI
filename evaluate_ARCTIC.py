"""
Evaluate a trained InteractionGNN checkpoint on the ARCTIC val split.

By default the model is fed ground-truth hand vertices (matching how it was
trained). To instead see how it performs on off-the-shelf HAMER hand-pose
estimates (aligned to the GT hand center -- see
dataset_ARCTIC.py:_align_HAMER_verts), set USE_HAMER_HANDS = True below.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset_ARCTIC import ArcticSequenceDataset, SameObjectBatchSampler
from model import InteractionGNN, apply_articulation_batch
from utils import extract_2d_bboxes, group_by_object, prepare_mano_edges, to_device

# The legacy evaluation always used HAMER vertices aligned to the GT hand centers.
USE_HAMER_HANDS = True


def create_pv_mesh(verts, faces, scalars=None):
    """Builds a PyVista mesh from vertices + triangular faces (optional -- see visualize_hands_pv)."""
    import numpy as np
    import pyvista as pv
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    faces_flat = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, faces_flat)
    if scalars is not None:
        pv_mesh["contact"] = scalars
    return pv_mesh


def visualize_hands_pv(gt_left_np, gt_right_np, hamer_left_np, hamer_right_np,
                        faces_left, faces_right, save_path=None):
    """
    Optional 2x2 GT-vs-HAMER hand comparison viewer (requires `pyvista`,
    not a hard dependency of the rest of this script). Not called by
    default -- call manually if you want to inspect a specific frame.
    """
    import numpy as np
    import pyvista as pv

    plotter = pv.Plotter(shape=(2, 2), window_size=(1800, 1200))

    gt_left_np, gt_right_np = np.asarray(gt_left_np), np.asarray(gt_right_np)
    hamer_left_np, hamer_right_np = np.asarray(hamer_left_np), np.asarray(hamer_right_np)
    faces_left, faces_right = np.asarray(faces_left), np.asarray(faces_right)

    meshes = [
        create_pv_mesh(gt_left_np, faces_left),
        create_pv_mesh(hamer_left_np, faces_left),
        create_pv_mesh(gt_right_np, faces_right),
        create_pv_mesh(hamer_right_np, faces_right),
    ]
    titles = ["GT Left", "HAMER Left", "GT Right", "HAMER Right"]

    for i, mesh in enumerate(meshes):
        plotter.subplot(i // 2, i % 2)
        plotter.add_text(titles[i], font_size=14)
        plotter.add_mesh(mesh, color="lightcoral" if "Left" in titles[i] else "lightblue",
                          smooth_shading=True, specular=0.3)
        plotter.view_isometric()
        plotter.add_axes()

    plotter.link_views()
    if save_path:
        plotter.show(screenshot=save_path)
    else:
        plotter.show()


def main():
    batch_size = 1
    seq_len = 1
    device = torch.device(os.environ.get("EGOPHI_EVAL_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = os.environ.get(
        "EGOPHI_EVAL_OUTPUT_DIR",
        os.path.join(config.PROJECT_ROOT, "evaluation_results", "arctic"),
    )
    max_samples = int(os.environ.get("EGOPHI_EVAL_MAX_SAMPLES", "0"))

    model = InteractionGNN(d_model=512, n_head=4, num_pose_blocks=4, num_force_blocks=4,
                            backbone_feature_dim=768, num_pose_iterations=3)
    model = model.to(device)

    checkpoint_path = os.environ.get(
        "EGOPHI_EVAL_CHECKPOINT",
        config.LEGACY_CHECKPOINT_PATH if os.path.isfile(config.LEGACY_CHECKPOINT_PATH)
        else config.BEST_CHECKPOINT_PATH,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.eval()
    print('MODEL LOADED')

    dataset = ArcticSequenceDataset(
        images_root=config.IMAGES_ROOT,
        segment_root=config.SEGMENT_ROOT,
        contacts_root=config.CONTACTS_ROOT,
        force_root=config.FORCE_ROOT,
        distances_root=config.DISTANCES_ROOT,
        processed_seqs_root=config.PROCESSED_SEQS_ROOT,
        mesh_root=config.MESH_ROOT,
        GT_mano_root=config.GT_MANO_ROOT,
        hamer_mano_root=config.HAMER_MANO_ROOT,
        load_hamer=False,
        cameras=[0],
        sequence_length=seq_len,
        overlap=0,
        participants=['s05'],  # only 's05' is used below (val) -- skip indexing everyone
                                # else. NOTE: if you fix the test-split bug below to use a
                                # different participant, add it here too or it'll index to nothing.
        device=None,
    )

    print('DATASET LOADED')
    val_indices = []
    test_indices = []
    for i, sample in enumerate(dataset.samples):
        participant = sample['participant']
        if participant == 's05':
            val_indices.append(i)
        # NOTE: this participant check duplicates the val one above, so
        # test_indices is always empty and test_loader (below) is unused.
        # Left as-is from the original script -- set a different held-out
        # participant here if you want an actual test split.
        elif participant == 's05':
            test_indices.append(i)

    # NOTE: the legacy evaluator's per-object frame window (which truncates each
    # object's sequence by the GLOBAL invalid-frame count, not a per-object one)
    # is now replicated inside ArcticSequenceDataset._index_sequences itself, so
    # dataset.samples already has the right length per object and no further
    # filtering is needed here.
    val_batches = group_by_object(val_indices, dataset)
    test_batches = group_by_object(test_indices, dataset)

    val_sampler = SameObjectBatchSampler(val_batches, batch_size=batch_size, shuffle=False)
    test_sampler = SameObjectBatchSampler(test_batches, batch_size=batch_size, shuffle=False)

    num_workers = int(os.environ.get("EGOPHI_EVAL_WORKERS", "8"))
    val_loader = DataLoader(dataset, batch_sampler=val_sampler, num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_sampler=test_sampler, num_workers=num_workers)  # noqa: F841

    hand_edges_left, hand_edges_right, _, _ = prepare_mano_edges()

    for sample_count, batch in enumerate(tqdm(val_loader, desc="Evaluating"), start=1):
        proc_seqs_path = batch['proc_seqs_path'][0]
        filename = os.path.basename(proc_seqs_path)[:-4]

        rgb_seq = batch['images'].squeeze(2).squeeze(1)
        object_seq = batch['obj_segmentation'].squeeze(2).squeeze(1)[:, 0]

        obj_template_vert = batch['obj_vert_template'].squeeze(1)
        obj_template_edges = batch['obj_edges_template'].squeeze(1)
        obj_part_ids = batch['obj_part_ids'].squeeze(1)
        pivot = batch['pivot'].squeeze(1)
        arti = batch['arti'].squeeze(1)
        K_ego = batch['K_ego'].squeeze(1)
        s = batch['s'].squeeze(1)
        if USE_HAMER_HANDS:
            vert_left = batch['hamer_left_aligned'].squeeze(1).to(device)
            vert_right = batch['hamer_right_aligned'].squeeze(1).to(device)
        else:
            vert_left = batch['left_verts'].squeeze(1).to(device)
            vert_right = batch['right_verts'].squeeze(1).to(device)
        H_orig, W_orig = 2000, 2800

        x_stack = rgb_seq
        verts_l_center = torch.mean(vert_left, dim=1)
        verts_r_center = torch.mean(vert_right, dim=1)
        object_bboxes = extract_2d_bboxes(object_seq)

        verts_articulated = apply_articulation_batch(
            vertices=obj_template_vert, part_ids=obj_part_ids, part_id_to_move=1,
            radians=arti, pivot=pivot, axis=None)

        (x_stack, verts_l_center, verts_r_center, vert_left, vert_right,
         object_bboxes, obj_template_vert, verts_articulated, obj_template_edges,
         s, K_ego) = to_device(
            device, x_stack, verts_l_center, verts_r_center, vert_left, vert_right,
            object_bboxes, obj_template_vert, verts_articulated, obj_template_edges,
            s, K_ego,
        )

        (pred_contact_left, pred_contact_right, pred_contact_obj,
         pred_left_force_magnitude, pred_right_force_magnitude,
         pred_obj_force_magnitude,
         pred_l_force_vector, pred_r_force_vector, pred_o_force_vector,
         all_pred_trans, all_pred_rot,
         final_pred_trans, final_pred_rot,
         verts_o_pred) = model(
            x_stack, vert_left, vert_right, verts_l_center, verts_r_center,
            verts_articulated, s, object_bboxes, K_ego,
            hand_edges_left, hand_edges_right, obj_template_edges[0],
            H_orig, W_orig,
        )

        if isinstance(batch['img_path'], (list, tuple)):
            img_path = batch['img_path'][0]
        else:
            img_path = batch['img_path']
        frame_id = Path(img_path).stem

        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        pred_left_force_magnitude = torch.sigmoid(pred_left_force_magnitude)
        pred_right_force_magnitude = torch.sigmoid(pred_right_force_magnitude)
        pred_obj_force_magnitude = torch.sigmoid(pred_obj_force_magnitude)

        pred_contact_left = torch.sigmoid(pred_contact_left)
        pred_contact_right = torch.sigmoid(pred_contact_right)
        pred_contact_obj = torch.sigmoid(pred_contact_obj)

        pred_left_force_vector = pred_l_force_vector / (pred_l_force_vector.norm(dim=-1, keepdim=True) + 1e-8)
        pred_right_force_vector = pred_r_force_vector / (pred_r_force_vector.norm(dim=-1, keepdim=True) + 1e-8)
        pred_obj_force_vector = pred_o_force_vector / (pred_o_force_vector.norm(dim=-1, keepdim=True) + 1e-8)

        left_contact_pred_batch = pred_contact_left.reshape(seq_len, -1).cpu()
        right_contact_pred_batch = pred_contact_right.reshape(seq_len, -1).cpu()
        obj_contact_pred_batch = pred_contact_obj.reshape(seq_len, -1).cpu()

        obj_force_vec_batch = pred_obj_force_vector.reshape(seq_len, -1, 3).cpu()
        left_force_vec_batch = pred_left_force_vector.reshape(seq_len, -1, 3).cpu()
        right_force_vec_batch = pred_right_force_vector.reshape(seq_len, -1, 3).cpu()

        obj_force_mag_batch = pred_obj_force_magnitude.reshape(seq_len, -1).cpu()
        left_force_mag_batch = pred_left_force_magnitude.reshape(seq_len, -1).cpu()
        right_force_mag_batch = pred_right_force_magnitude.reshape(seq_len, -1).cpu()

        # Predictions
        torch.save(left_force_vec_batch, os.path.join(save_dir, f'left_force_vec_pred_{frame_id}.pt'))
        torch.save(right_force_vec_batch, os.path.join(save_dir, f'right_force_vec_pred_{frame_id}.pt'))
        torch.save(obj_force_vec_batch, os.path.join(save_dir, f'obj_force_vec_pred_{frame_id}.pt'))

        torch.save(left_force_mag_batch, os.path.join(save_dir, f'left_force_mag_pred_{frame_id}.pt'))
        torch.save(right_force_mag_batch, os.path.join(save_dir, f'right_force_mag_pred_{frame_id}.pt'))
        torch.save(obj_force_mag_batch, os.path.join(save_dir, f'obj_force_mag_pred_{frame_id}.pt'))

        torch.save(left_contact_pred_batch, os.path.join(save_dir, f'left_contact_pred_{frame_id}.pt'))
        torch.save(right_contact_pred_batch, os.path.join(save_dir, f'right_contact_pred_{frame_id}.pt'))
        torch.save(obj_contact_pred_batch, os.path.join(save_dir, f'obj_contact_pred_{frame_id}.pt'))

        if max_samples and sample_count >= max_samples:
            break


if __name__ == '__main__':
    main()