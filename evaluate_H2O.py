"""
Evaluate a trained InteractionGNN checkpoint on the H2O dataset.

H2O has no ground-truth hand mesh, so the model is always fed
HAMER-estimated hand vertices (aligned to the dataset's own hand-position
estimate -- see dataset_H2O.py:_load_obj_R_T_arti), unlike EVAL_ARCTIC.py
which defaults to GT hands.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config
from dataset_H2O import H2OSequenceDataset
from model import InteractionGNN
from utils import extract_2d_bboxes, prepare_mano_edges, to_device


def main():
    seq_len = 1
    device = torch.device(os.environ.get("EGOPHI_EVAL_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = os.environ.get(
        "EGOPHI_EVAL_OUTPUT_DIR",
        os.path.join(config.PROJECT_ROOT, "evaluation_results", "h2o"),
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

    dataset = H2OSequenceDataset(
        images_root=config.H2O_IMAGES_ROOT,
        contacts_root=config.H2O_CONTACTS_ROOT,
        force_root=config.H2O_FORCE_ROOT,
        cameras=[0],
        sequence_length=seq_len,
        overlap=0,
        subjects=['subject4_ego'],  # only subject used below -- skip indexing everyone else
    )
    print(f"Dataset size: {len(dataset)}")

    val_indices = [
        i for i, sample in enumerate(dataset.samples)
        if sample.get('participant') == 'subject4_ego'
    ]
    val_dataset = Subset(dataset, val_indices)
    num_workers = int(os.environ.get("EGOPHI_EVAL_WORKERS", "8"))
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    hand_edges_left, hand_edges_right, _, _ = prepare_mano_edges()

    for sample_count, batch in enumerate(tqdm(val_loader, desc="Evaluating"), start=1):
        participant = batch['participant'][0] if isinstance(batch['participant'], (list, tuple)) else batch['participant']
        if participant != 'subject4_ego':
            print('SKIPPING NON SUBJECT4_EGO')
            continue

        rgb_seq = batch['images'].squeeze(0)
        object_seq = batch['obj_segmentation'].squeeze(0).squeeze(0).to(device)
        object_bboxes = extract_2d_bboxes(object_seq).to(device)
        s = batch['scale'].squeeze(0).to(device)
        obj_template_vert = batch['obj_vert_template'].to(device)
        K_ego = batch['K_ego'].squeeze(0)
        obj_template_edges = batch['obj_edges_template'].squeeze(0).permute(1, 0)

        vert_left = batch['hamer_left_aligned'].reshape(1, 778, 3)
        vert_right = batch['hamer_right_aligned'].reshape(1, 778, 3)

        filename = batch['object'][0].replace('/', '_')

        img_path = batch['img_path']
        if isinstance(img_path, (list, tuple)):
            img_path = img_path[0]
        frame_id = Path(img_path).stem

        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        files_to_check = [
            f'left_force_mag_pred_{frame_id}.pt',
            f'right_force_mag_pred_{frame_id}.pt',
            f'obj_force_mag_pred_{frame_id}.pt',
            f'left_contact_pred_{frame_id}.pt',
            f'right_contact_pred_{frame_id}.pt',
            f'obj_contact_pred_{frame_id}.pt',
            f'left_force_vec_pred_{frame_id}.pt',
            f'right_force_vec_pred_{frame_id}.pt',
            f'obj_force_vec_pred_{frame_id}.pt',
        ]
        if all(os.path.exists(os.path.join(save_dir, f)) for f in files_to_check):
            print(f"Skipping frame {frame_id} for {filename}, all outputs already exist.")
            continue

        verts_l_center = torch.mean(vert_left, dim=1).to(device)
        verts_r_center = torch.mean(vert_right, dim=1).to(device)

        (rgb_seq, vert_left, vert_right, obj_template_edges, obj_template_vert, K_ego) = to_device(
            device, rgb_seq, vert_left, vert_right, obj_template_edges, obj_template_vert, K_ego,
        )

        H_orig, W_orig = 720, 1280

        (pred_contact_left, pred_contact_right, pred_contact_obj,
         pred_left_force_magnitude, pred_right_force_magnitude,
         pred_obj_force_magnitude,
         pred_l_force_vector, pred_r_force_vector, pred_o_force_vector,
         all_pred_trans, all_pred_rot,
         final_pred_trans, final_pred_rot,
         verts_o_pred) = model(
            rgb_seq, vert_left, vert_right, verts_l_center, verts_r_center,
            obj_template_vert, s, object_bboxes, K_ego,
            hand_edges_left, hand_edges_right, obj_template_edges,
            H_orig, W_orig,
        )

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