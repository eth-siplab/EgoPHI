"""
Training entry point for InteractionGNN (hand-object pose + contact-force
estimation on ARCTIC).

Data and checkpoint paths are defined in config.py.
"""

import os
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from manopth.manolayer import ManoLayer
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset_ARCTIC import ArcticSequenceDataset, SameObjectBatchSampler
from model import InteractionGNN, apply_articulation_batch
from utils import (
    RegLoss, RegLossForce, SmoothRegLoss, VCBLoss,
    extract_2d_bboxes, group_by_object, prepare_mano_edges, render_contact_hand,
    render_force_colored_mesh_torch, render_force_hand, rotation_loss, to_device,
    translation_loss, weighted_bce_loss, weighted_global_l2_loss,
)

# Anomaly detection is expensive (adds synchronous checks around every
# backward op) and was only ever meant to be flipped on temporarily while
# tracking down a NaN/Inf gradient -- keep it off for normal training.
torch.autograd.set_detect_anomaly(False)


def log_wandb_visuals(prefix, epoch, T, batch, world_verts, verts_o_pred, pred_obj_force_magnitude, force_object,
                       verts_l, verts_r, faces_l, faces_r,
                       pred_left_force_magnitude, force_left, pred_right_force_magnitude, force_right,
                       pred_contact_left, contact_left, pred_contact_right, contact_right, rgb_seq):
    """
    Renders and wandb-logs the mesh/force/contact visualizations for one
    batch, under `{prefix}/...` keys. Shared by train_one_epoch and evaluate.
    """
    B = world_verts.shape[0]
    frame_ids = [0]

    rendered_magn = render_force_colored_mesh_torch(100*verts_o_pred.reshape(B, T, -1, 3), batch['obj_faces_template'], pred_obj_force_magnitude.reshape(B, T, -1, 1), frame_ids)
    rendered_GT_magn = render_force_colored_mesh_torch(100*world_verts.reshape(B, T, -1, 3), batch['obj_faces_template'], force_object.reshape(B, T, -1, 1), frame_ids)

    rendered_left = render_force_hand(verts_l.reshape(-1, 3), faces_l, pred_left_force_magnitude.reshape(B, T, -1, 1), frame_ids)
    rendered_GT_left = render_force_hand(verts_l.reshape(-1, 3), faces_l, force_left.reshape(B, T, -1, 1), frame_ids)
    rendered_right = render_force_hand(verts_r.reshape(-1, 3), faces_r, pred_right_force_magnitude.reshape(B, T, -1, 1), frame_ids)
    rendered_GT_right = render_force_hand(verts_r.reshape(-1, 3), faces_r, force_right.reshape(B, T, -1, 1), frame_ids)

    rendered_contact_left = render_contact_hand(verts_l, faces_l, torch.sigmoid(pred_contact_left).squeeze(-1), frame_ids)
    rendered_GT_contact_left = render_contact_hand(verts_l.reshape(-1, 3), faces_l, contact_left, frame_ids)
    rendered_contact_right = render_contact_hand(verts_r, faces_r, torch.sigmoid(pred_contact_right).squeeze(-1), frame_ids)
    rendered_GT_contact_right = render_contact_hand(verts_r.reshape(-1, 3), faces_r, contact_right, frame_ids)

    for t, (front, back) in rendered_magn.items():
        wandb.log({
            f"{prefix}/FORCE_PRED_MAGNITUDE/frame_{t}_front": wandb.Image(front),
            f"{prefix}/FORCE_PRED_MAGNITUDE/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    for t, (front, back) in rendered_GT_magn.items():
        wandb.log({
            f"{prefix}/FORCE_GT_MAGNITUDE/frame_{t}_front": wandb.Image(front),
            f"{prefix}/FORCE_GT_MAGNITUDE/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    # LEFT
    for t, (front, back) in rendered_left.items():
        wandb.log({
            f"{prefix}/FORCE_PRED_LEFT/frame_{t}_front": wandb.Image(front),
            f"{prefix}/FORCE_PRED_LEFT/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    for t, (front, back) in rendered_GT_left.items():
        wandb.log({
            f"{prefix}/FORCE_GT_LEFT/frame_{t}_front": wandb.Image(front),
            f"{prefix}/FORCE_GT_LEFT/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    for t, (front, back) in rendered_contact_left.items():
        wandb.log({
            f"{prefix}/CONTACT_PRED_LEFT/frame_{t}_front": wandb.Image(front),
            f"{prefix}/CONTACT_PRED_LEFT/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    for t, (front, back) in rendered_GT_contact_left.items():
        wandb.log({
            f"{prefix}/CONTACT_GT_LEFT/frame_{t}_front": wandb.Image(front),
            f"{prefix}/CONTACT_GT_LEFT/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    # RIGHT
    for t, (front, back) in rendered_right.items():
        wandb.log({
            f"{prefix}/FORCE_PRED_RIGHT/frame_{t}_front": wandb.Image(front),
            f"{prefix}/FORCE_PRED_RIGHT/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    for t, (front, back) in rendered_GT_right.items():
        wandb.log({
            f"{prefix}/FORCE_GT_RIGHT/frame_{t}_front": wandb.Image(front),
            f"{prefix}/FORCE_GT_RIGHT/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    for t, (front, back) in rendered_contact_right.items():
        wandb.log({
            f"{prefix}/CONTACT_PRED_RIGHT/frame_{t}_front": wandb.Image(front),
            f"{prefix}/CONTACT_PRED_RIGHT/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    for t, (front, back) in rendered_GT_contact_right.items():
        wandb.log({
            f"{prefix}/CONTACT_GT_RIGHT/frame_{t}_front": wandb.Image(front),
            f"{prefix}/CONTACT_GT_RIGHT/frame_{t}_back": wandb.Image(back),
            "epoch": epoch + 1
        })

    for frame_id in frame_ids:
        rgb_img = (rgb_seq[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rgb_img = Image.fromarray(rgb_img)
        wandb.log({f"{prefix}/rgb/frame_{frame_id}": wandb.Image(rgb_img),
                "epoch": epoch + 1})


def train_one_epoch(use_wandb, model, dataloader, optimizer, device, reg_loss_force, vcb_loss, reg_loss, smooth_loss, is_distributed,
                    hand_edges_left, hand_edges_right, hand_normals_left, hand_normals_right, num_vertices_to_render, epoch):
    model.train()
    total_loss = 0.0
    total_loss_contact = 0.0
    total_loss_force_mag = 0.0
    total_loss_force_vec = 0.0
    total_loss_rot = 0.0
    total_loss_trans = 0.0
    total_loss_vertices = 0.0

    mano_layer_right = ManoLayer(flat_hand_mean=False, side='right', use_pca=False, mano_root=config.MANO_ROOT)
    mano_layer_left = ManoLayer(flat_hand_mean=False, side='left', use_pca=False, mano_root=config.MANO_ROOT)
    
    faces_r = mano_layer_right.th_faces
    faces_l = mano_layer_left.th_faces
    
    verts_r = mano_layer_right.th_v_template.squeeze(0)
    verts_l = mano_layer_left.th_v_template.squeeze(0)

    for batch in tqdm(dataloader, desc="Training"):
        # --- Unpack batch ---
        rgb_seq = batch['images'].squeeze(2).squeeze(1)
        object_seq = batch['obj_segmentation'].squeeze(2).squeeze(1)[:,0]

        # GT hand vertices
        vert_left = batch['left_verts'].squeeze(1)
        vert_right = batch['right_verts'].squeeze(1)

        obj_template_vert = batch['obj_vert_template'].squeeze(1)
        obj_template_faces = batch['obj_faces_template'].squeeze(1)
        obj_template_edges = batch['obj_edges_template'].squeeze(1)
        obj_edges_weight = batch['obj_edges_weight'].squeeze(1) 
        obj_part_ids = batch['obj_part_ids'].squeeze(1) 
        pivot = batch['pivot'].squeeze(1)
        obj_normals = batch['vertex_normals'].squeeze(1)
        contact_left = batch['contacts']['left'].squeeze(3).squeeze(1)
        contact_right = batch['contacts']['right'].squeeze(3).squeeze(1)
        contact_object = batch['contacts']['object'].squeeze(3).squeeze(1)
        force_left = batch['forces']['left']
        force_right = batch['forces']['right']
        force_object = batch['forces']['object']
        force_left_vector = batch['force_vector']['left']
        force_right_vector = batch['force_vector']['right']
        force_object_vector = batch['force_vector']['object']
        world2ego = batch['world2ego'].squeeze(1)
        K_ego = batch['K_ego'].squeeze(1) 
        world_verts = batch['world_verts'].squeeze(1)
        R = batch['R'].squeeze(1)
        t = batch['T'].squeeze(1)
        s = batch['s'].squeeze(1)
        arti = batch['arti'].squeeze(1)
        bbox3d = batch['bbox3d'].squeeze(1)
        T = 1 # Assuming sequence length processing is 1

        # --- Input preparation ---
        # x_stack: [B, 3, H, W]
        x_stack = rgb_seq

        # Create hand centers (using mean of vertices)
        verts_l_center = torch.mean(vert_left, dim=1) # [B, 3]
        verts_r_center = torch.mean(vert_right, dim=1) # [B, 3]

        object_bboxes = extract_2d_bboxes(object_seq)

        # Articulate object
        verts_articulated = apply_articulation_batch(
            vertices=obj_template_vert,
            part_ids=obj_part_ids,
            part_id_to_move=1,
            radians=arti,
            pivot=pivot,
            axis=None)

        (x_stack, verts_l_center, verts_r_center,
         rgb_seq, object_seq, verts_articulated,
         vert_left, vert_right, object_bboxes,
         obj_template_vert, obj_normals, obj_template_edges, obj_edges_weight, bbox3d, obj_part_ids, pivot,
         contact_left, contact_right, contact_object, force_left, force_right, force_object,
         force_left_vector, force_right_vector, force_object_vector,
         R, t, arti, s, world_verts, K_ego) = to_device(
            device, x_stack, verts_l_center, verts_r_center,
            rgb_seq, object_seq, verts_articulated,
            vert_left, vert_right, object_bboxes,
            obj_template_vert, obj_normals, obj_template_edges, obj_edges_weight, bbox3d, obj_part_ids, pivot,
            contact_left, contact_right, contact_object, force_left, force_right, force_object,
            force_left_vector, force_right_vector, force_object_vector,
            R, t, arti, s, world_verts, K_ego
        )

        optimizer.zero_grad()

        B, _, H, W = rgb_seq.shape
        H_orig, W_orig = 2000, 2800

        (
            pred_contact_left, pred_contact_right, pred_contact_obj,      # 1-3
            pred_left_force_magnitude, pred_right_force_magnitude,       # 4-5
            pred_obj_force_magnitude,                                   # 6
            pred_l_force_vector, pred_r_force_vector, pred_o_force_vector, # 7-9
            all_pred_trans, all_pred_rot,                               # 10-11 (Iterative poses)
            final_pred_trans, final_pred_rot,                           # 12-13 (Final pose)
            verts_o_pred                                                # 14
        ) = model(
            x_stack,            # [B, 6, H, W]
            vert_left,          # [B, N_hand, 3]
            vert_right,         # [B, N_hand, 3]
            verts_l_center,     # [B, 3]
            verts_r_center,     # [B, 3]
            verts_articulated,  # [B, N_obj, 3]
            s,                  # [B, 1]
            object_bboxes,      # [B, 4]
            K_ego,              # [B, 3, 3]
            hand_edges_left,    # [2, E_hand]
            hand_edges_right,   # [2, E_hand]
            obj_template_edges[0], # [2, E_obj]
            H_orig, W_orig
        )
        
        # --- Contact and force losses ---
        pred_left_force_magnitude = torch.sigmoid(pred_left_force_magnitude)
        pred_right_force_magnitude = torch.sigmoid(pred_right_force_magnitude)
        pred_obj_force_magnitude = torch.sigmoid(pred_obj_force_magnitude)

        pred_l_force_vector = pred_l_force_vector / (pred_l_force_vector.norm(dim=-1, keepdim=True) + 1e-8)
        pred_r_force_vector = pred_r_force_vector / (pred_r_force_vector.norm(dim=-1, keepdim=True) + 1e-8)
        pred_o_force_vector = pred_o_force_vector / (pred_o_force_vector.norm(dim=-1, keepdim=True) + 1e-8)

        B = contact_left.shape[0]

        left_loss_force = weighted_global_l2_loss(pred_l_force_vector.reshape(B, T, -1, 3).float(), force_left_vector.reshape(B, T, -1, 3).float(), contact_left.reshape(B, T, -1, 1).float())
        right_loss_force = weighted_global_l2_loss(pred_r_force_vector.reshape(B, T, -1, 3).float(), force_right_vector.reshape(B, T, -1, 3).float(), contact_right.reshape(B, T, -1, 1).float())
        object_loss_force = weighted_global_l2_loss(pred_o_force_vector.reshape(B, T, -1, 3).float(), force_object_vector.reshape(B, T, -1, 3).float(), contact_object.reshape(B, T, -1, 1).float())
        loss_force_vec = (left_loss_force + right_loss_force + object_loss_force)/4
        
        reg_loss_left_magn = reg_loss_force(pred_left_force_magnitude.float().reshape(-1,778), obj_type=batch['object'][0].split("_")[0], hand='left')
        reg_loss_right_magn = reg_loss_force(pred_right_force_magnitude.float().reshape(-1,778), obj_type=batch['object'][0].split("_")[0], hand='right')
        reg_loss_object_magn = reg_loss_force(pred_obj_force_magnitude.float().reshape(B*T,-1), obj_type=batch['object'][0].split("_")[0], hand=None)

        left_loss_force_magn = weighted_global_l2_loss(pred_left_force_magnitude.reshape(B, T, -1, 1).float(), force_left.unsqueeze(-1).reshape(B, T, -1, 1).float(), contact_left.reshape(B, T, -1, 1).float())
        right_loss_force_magn = weighted_global_l2_loss(pred_right_force_magnitude.reshape(B, T, -1, 1).float(), force_right.unsqueeze(-1).reshape(B, T, -1, 1).float(), contact_right.reshape(B, T, -1, 1).float())
        object_loss_force_magn = weighted_global_l2_loss(pred_obj_force_magnitude.reshape(B, T, -1, 1).float(), force_object.unsqueeze(-1).reshape(B, T, -1, 1).float(), contact_object.reshape(B, T, -1, 1).float())
        loss_force_mag = 10*(left_loss_force_magn + right_loss_force_magn + object_loss_force_magn+ reg_loss_left_magn + reg_loss_right_magn + reg_loss_object_magn)/17
        
        contact_vcb_left = vcb_loss(pred_contact_left.float().reshape(-1,778), contact_left.float().reshape(-1,778), epoch=epoch) * 8
        contact_vcb_right = vcb_loss(pred_contact_right.float().reshape(-1,778), contact_right.float().reshape(-1,778), epoch=epoch) * 8
        reg_loss_left = reg_loss(pred_contact_left.float().reshape(-1,778), obj_type=None) * 8
        reg_loss_right = reg_loss(pred_contact_right.float().reshape(-1,778), obj_type=None) * 8
        reg_loss_object = reg_loss(pred_contact_obj.float().reshape(B*T,-1), obj_type=batch['object'][0].split("_")[0]) * 12
        smooth_loss_left = smooth_loss(pred_contact_left.float().reshape(-1,778), faces_l.view(-1, 3)) * 10
        smooth_loss_right = smooth_loss(pred_contact_right.float().reshape(-1,778), faces_r.view(-1, 3)) * 10 
        smooth_loss_object = smooth_loss(pred_contact_obj.float().reshape(B*T,-1), obj_template_faces.view(-1, 3)) * 1000
        
        left_contact_loss = weighted_bce_loss(pred_contact_left.squeeze(-1).float(), contact_left.squeeze(-1).float()) + contact_vcb_left + reg_loss_left + smooth_loss_left
        right_contact_loss = weighted_bce_loss(pred_contact_right.squeeze(-1).float(), contact_right.squeeze(-1).float()) + contact_vcb_right + reg_loss_right + smooth_loss_right
        obj_contact_loss = weighted_bce_loss(pred_contact_obj.squeeze(-1).float(), contact_object.squeeze(-1).float()) + smooth_loss_object + reg_loss_object
        loss_contact = 10*(left_contact_loss + right_contact_loss + obj_contact_loss)/100

        # --- Iterative pose losses ---
        loss_verts = F.mse_loss(verts_o_pred, world_verts) * 10 * 20
        
        # New Iterative Pose Losses
        num_iters = all_pred_trans.shape[1]
        
        # Expand GT to match iterative predictions
        gt_t_expanded = t.unsqueeze(1).expand_as(all_pred_trans)
        gt_R_expanded = R.unsqueeze(1).expand_as(all_pred_rot)
        
        # Apply losses with linearly increasing weight (optional, but good)
        t_losses = []
        r_losses = []
        for i in range(num_iters):
            # Weight is (i+1) / num_iters, e.g., [0.33, 0.66, 1.0] for 3 iters
            weight = (i + 1) / num_iters 
            t_losses.append(
                weight * translation_loss(all_pred_trans[:, i].unsqueeze(1), gt_t_expanded[:, i])
            )
            r_losses.append(
                weight * rotation_loss(all_pred_rot[:, i].unsqueeze(1), gt_R_expanded[:, i])
            )
        
        # Average the weighted losses
        loss_t = torch.mean(torch.stack(t_losses)) * 10 * 12
        loss_r = torch.mean(torch.stack(r_losses)) / 1.4

        # --- Total loss and backprop ---
        loss = loss_contact + loss_force_mag + loss_force_vec + loss_verts + loss_t + loss_r

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Bookkeeping ---
        total_loss += loss.detach().item()
        total_loss_contact += loss_contact.detach().item()
        total_loss_force_mag += loss_force_mag.detach().item()   
        total_loss_force_vec += loss_force_vec.detach().item()
        total_loss_rot += loss_r.detach().item()
        total_loss_trans += loss_t.detach().item()
        total_loss_vertices += loss_verts.detach().item()
            
    if use_wandb and ((not is_distributed) or (dist.get_rank() == 0)):   
        with torch.no_grad():
            log_wandb_visuals("train", epoch, T, batch, world_verts, verts_o_pred, pred_obj_force_magnitude, force_object,
                               verts_l, verts_r, faces_l, faces_r,
                               pred_left_force_magnitude, force_left, pred_right_force_magnitude, force_right,
                               pred_contact_left, contact_left, pred_contact_right, contact_right, rgb_seq)
    
    avg_loss = total_loss / len(dataloader)

    log_dict = {
                        "TRAIN Total train loss": avg_loss,
                        "TRAIN Total contact loss" : total_loss_contact / len(dataloader),
                        "TRAIN Total force magn loss": total_loss_force_mag / len(dataloader),
                        "TRAIN Total force vec loss": total_loss_force_vec / len(dataloader),
                        "TRAIN Total rot loss": total_loss_rot / len(dataloader),
                        "TRAIN Total trans loss": total_loss_trans / len(dataloader),
                        "TRAIN Total vertices loss": total_loss_vertices / len(dataloader)
                    }

    if use_wandb and ((not is_distributed) or (dist.get_rank() == 0)): 
        wandb.log(log_dict)

    torch.cuda.empty_cache()

    return avg_loss


@torch.no_grad()
def evaluate(use_wandb, model, dataloader, device, reg_loss_force, vcb_loss, reg_loss, smooth_loss, is_distributed, hand_edges_left, hand_edges_right, hand_normals_left, hand_normals_right, num_vertices_to_render, epoch, val):
    model.eval()
    total_loss = 0.0
    total_loss_contact = 0.0
    total_loss_force_mag = 0.0
    total_loss_force_vec = 0.0
    total_loss_rot = 0.0
    total_loss_trans = 0.0
    total_loss_vertices = 0.0


    # MANO hands
    mano_layer_right = ManoLayer(flat_hand_mean=False, side='right', use_pca=False, mano_root=config.MANO_ROOT)
    mano_layer_left = ManoLayer(flat_hand_mean=False, side='left', use_pca=False, mano_root=config.MANO_ROOT)
    
    faces_r = mano_layer_right.th_faces
    faces_l = mano_layer_left.th_faces
    
    # Vertices of the canonical hand mesh
    verts_r = mano_layer_right.th_v_template.squeeze(0)
    verts_l = mano_layer_left.th_v_template.squeeze(0)

    for batch in tqdm(dataloader, desc="Evaluation"):
        # Visual
        rgb_seq = batch['images'].squeeze(2).squeeze(1)
        object_seq = batch['obj_segmentation'].squeeze(2).squeeze(1)[:,0]

        # GT hand vertices
        vert_left = batch['left_verts'].squeeze(1)
        vert_right = batch['right_verts'].squeeze(1)

        obj_template_vert = batch['obj_vert_template'].squeeze(1)
        obj_template_faces = batch['obj_faces_template'].squeeze(1)
        obj_template_edges = batch['obj_edges_template'].squeeze(1)
        obj_edges_weight = batch['obj_edges_weight'].squeeze(1) 
        obj_part_ids = batch['obj_part_ids'].squeeze(1) 
        pivot = batch['pivot'].squeeze(1)
        obj_normals = batch['vertex_normals'].squeeze(1)
        contact_left = batch['contacts']['left'].squeeze(3).squeeze(1)
        contact_right = batch['contacts']['right'].squeeze(3).squeeze(1)
        contact_object = batch['contacts']['object'].squeeze(3).squeeze(1)
        force_left = batch['forces']['left']
        force_right = batch['forces']['right']
        force_object = batch['forces']['object']
        force_left_vector = batch['force_vector']['left']
        force_right_vector = batch['force_vector']['right']
        force_object_vector = batch['force_vector']['object']
        world2ego = batch['world2ego'].squeeze(1)
        K_ego = batch['K_ego'].squeeze(1) 
        world_verts = batch['world_verts'].squeeze(1)
        R = batch['R'].squeeze(1)
        t = batch['T'].squeeze(1)
        s = batch['s'].squeeze(1)
        arti = batch['arti'].squeeze(1)
        bbox3d = batch['bbox3d'].squeeze(1)
        T = 1 # Assuming sequence length processing is 1

        # --- Input preparation ---
        x_stack = rgb_seq

        # Create hand centers (using mean of vertices)
        verts_l_center = torch.mean(vert_left, dim=1) # [B, 3]
        verts_r_center = torch.mean(vert_right, dim=1) # [B, 3]

        object_bboxes = extract_2d_bboxes(object_seq)

        # Articulate object
        verts_articulated = apply_articulation_batch(
            vertices=obj_template_vert,
            part_ids=obj_part_ids,
            part_id_to_move=1,
            radians=arti,
            pivot=pivot,
            axis=None)

        (x_stack, verts_l_center, verts_r_center,
         rgb_seq, object_seq, verts_articulated,
         vert_left, vert_right, object_bboxes,
         obj_template_vert, obj_normals, obj_template_edges, obj_edges_weight, bbox3d, obj_part_ids, pivot,
         contact_left, contact_right, contact_object, force_left, force_right, force_object,
         force_left_vector, force_right_vector, force_object_vector,
         R, t, arti, s, world_verts, K_ego) = to_device(
            device, x_stack, verts_l_center, verts_r_center,
            rgb_seq, object_seq, verts_articulated,
            vert_left, vert_right, object_bboxes,
            obj_template_vert, obj_normals, obj_template_edges, obj_edges_weight, bbox3d, obj_part_ids, pivot,
            contact_left, contact_right, contact_object, force_left, force_right, force_object,
            force_left_vector, force_right_vector, force_object_vector,
            R, t, arti, s, world_verts, K_ego
        )

        B, _, H, W = rgb_seq.shape
        H_orig, W_orig = 2000, 2800

        (
            pred_contact_left, pred_contact_right, pred_contact_obj,      # 1-3
            pred_left_force_magnitude, pred_right_force_magnitude,       # 4-5
            pred_obj_force_magnitude,                                   # 6
            pred_l_force_vector, pred_r_force_vector, pred_o_force_vector, # 7-9
            all_pred_trans, all_pred_rot,                               # 10-11 (Iterative poses)
            final_pred_trans, final_pred_rot,                           # 12-13 (Final pose)
            verts_o_pred                                                # 14
        ) = model(
            x_stack,            # [B, 6, H, W]
            vert_left,          # [B, N_hand, 3]
            vert_right,         # [B, N_hand, 3]
            verts_l_center,     # [B, 3]
            verts_r_center,     # [B, 3]
            verts_articulated,  # [B, N_obj, 3]
            s,                  # [B, 1]
            object_bboxes,      # [B, 4]
            K_ego,              # [B, 3, 3]
            hand_edges_left,    # [2, E_hand]
            hand_edges_right,   # [2, E_hand]
            obj_template_edges[0], # [2, E_obj]
            H_orig, W_orig
        )
        
        # --- Contact and force losses ---
        pred_left_force_magnitude = torch.sigmoid(pred_left_force_magnitude)
        pred_right_force_magnitude = torch.sigmoid(pred_right_force_magnitude)
        pred_obj_force_magnitude = torch.sigmoid(pred_obj_force_magnitude)

        pred_l_force_vector = pred_l_force_vector / (pred_l_force_vector.norm(dim=-1, keepdim=True) + 1e-8)
        pred_r_force_vector = pred_r_force_vector / (pred_r_force_vector.norm(dim=-1, keepdim=True) + 1e-8)
        pred_o_force_vector = pred_o_force_vector / (pred_o_force_vector.norm(dim=-1, keepdim=True) + 1e-8)

        B = contact_left.shape[0]

        left_loss_force = weighted_global_l2_loss(pred_l_force_vector.reshape(B, T, -1, 3).float(), force_left_vector.reshape(B, T, -1, 3).float(), contact_left.reshape(B, T, -1, 1).float())
        right_loss_force = weighted_global_l2_loss(pred_r_force_vector.reshape(B, T, -1, 3).float(), force_right_vector.reshape(B, T, -1, 3).float(), contact_right.reshape(B, T, -1, 1).float())
        object_loss_force = weighted_global_l2_loss(pred_o_force_vector.reshape(B, T, -1, 3).float(), force_object_vector.reshape(B, T, -1, 3).float(), contact_object.reshape(B, T, -1, 1).float())
        loss_force_vec = (left_loss_force + right_loss_force + object_loss_force)/4
        
        reg_loss_left_magn = reg_loss_force(pred_left_force_magnitude.float().reshape(-1,778), obj_type=batch['object'][0].split("_")[0], hand='left')
        reg_loss_right_magn = reg_loss_force(pred_right_force_magnitude.float().reshape(-1,778), obj_type=batch['object'][0].split("_")[0], hand='right')
        reg_loss_object_magn = reg_loss_force(pred_obj_force_magnitude.float().reshape(B*T,-1), obj_type=batch['object'][0].split("_")[0], hand=None)

        left_loss_force_magn = weighted_global_l2_loss(pred_left_force_magnitude.reshape(B, T, -1, 1).float(), force_left.unsqueeze(-1).reshape(B, T, -1, 1).float(), contact_left.reshape(B, T, -1, 1).float())
        right_loss_force_magn = weighted_global_l2_loss(pred_right_force_magnitude.reshape(B, T, -1, 1).float(), force_right.unsqueeze(-1).reshape(B, T, -1, 1).float(), contact_right.reshape(B, T, -1, 1).float())
        object_loss_force_magn = weighted_global_l2_loss(pred_obj_force_magnitude.reshape(B, T, -1, 1).float(), force_object.unsqueeze(-1).reshape(B, T, -1, 1).float(), contact_object.reshape(B, T, -1, 1).float())
        loss_force_mag = 10*(left_loss_force_magn + right_loss_force_magn + object_loss_force_magn+ reg_loss_left_magn + reg_loss_right_magn + reg_loss_object_magn)/17
        
        contact_vcb_left = vcb_loss(pred_contact_left.float().reshape(-1,778), contact_left.float().reshape(-1,778), epoch=epoch) * 8
        contact_vcb_right = vcb_loss(pred_contact_right.float().reshape(-1,778), contact_right.float().reshape(-1,778), epoch=epoch) * 8
        reg_loss_left = reg_loss(pred_contact_left.float().reshape(-1,778), obj_type=None) * 8
        reg_loss_right = reg_loss(pred_contact_right.float().reshape(-1,778), obj_type=None) * 8
        reg_loss_object = reg_loss(pred_contact_obj.float().reshape(B*T,-1), obj_type=batch['object'][0].split("_")[0]) * 12
        smooth_loss_left = smooth_loss(pred_contact_left.float().reshape(-1,778), faces_l.view(-1, 3)) * 10
        smooth_loss_right = smooth_loss(pred_contact_right.float().reshape(-1,778), faces_r.view(-1, 3)) * 10 
        smooth_loss_object = smooth_loss(pred_contact_obj.float().reshape(B*T,-1), obj_template_faces.view(-1, 3)) * 1000
        
        left_contact_loss = weighted_bce_loss(pred_contact_left.squeeze(-1).float(), contact_left.squeeze(-1).float()) + contact_vcb_left + reg_loss_left + smooth_loss_left
        right_contact_loss = weighted_bce_loss(pred_contact_right.squeeze(-1).float(), contact_right.squeeze(-1).float()) + contact_vcb_right + reg_loss_right + smooth_loss_right
        obj_contact_loss = weighted_bce_loss(pred_contact_obj.squeeze(-1).float(), contact_object.squeeze(-1).float()) + smooth_loss_object + reg_loss_object
        loss_contact = 10*(left_contact_loss + right_contact_loss + obj_contact_loss)/100

        # --- Iterative pose losses ---
        loss_verts = F.mse_loss(verts_o_pred, world_verts) * 10 * 20
        
        # New Iterative Pose Losses
        num_iters = all_pred_trans.shape[1]
        
        # Expand GT to match iterative predictions
        gt_t_expanded = t.unsqueeze(1).expand_as(all_pred_trans)
        gt_R_expanded = R.unsqueeze(1).expand_as(all_pred_rot)
        
        # Apply losses with linearly increasing weight (optional, but good)
        t_losses = []
        r_losses = []
        for i in range(num_iters):
            # Weight is (i+1) / num_iters, e.g., [0.33, 0.66, 1.0] for 3 iters
            weight = (i + 1) / num_iters 
            t_losses.append(
                weight * translation_loss(all_pred_trans[:, i].unsqueeze(1), gt_t_expanded[:, i])
            )
            r_losses.append(
                weight * rotation_loss(all_pred_rot[:, i].unsqueeze(1), gt_R_expanded[:, i])
            )
        
        # Average the weighted losses
        loss_t = torch.mean(torch.stack(t_losses)) * 10 * 12
        loss_r = torch.mean(torch.stack(r_losses)) / 1.4

        # --- Total loss ---
        loss = loss_contact + loss_force_mag + loss_force_vec + loss_verts + loss_t + loss_r

        total_loss += loss.detach().item()
        total_loss_contact += loss_contact.detach().item()
        total_loss_force_mag += loss_force_mag.detach().item()   
        total_loss_force_vec += loss_force_vec.detach().item()
        total_loss_rot += loss_r.detach().item()
        total_loss_trans += loss_t.detach().item()
        total_loss_vertices += loss_verts.detach().item()

    avg_loss = total_loss / len(dataloader)

    if use_wandb and ((not is_distributed) or (dist.get_rank() == 0)):
        T = 1
        with torch.no_grad():
            if val:
                log_dict = {
                        "VAL Total train loss": avg_loss,
                        "VAL Total contact loss" : total_loss_contact / len(dataloader),
                        "VAL Total force magn loss": total_loss_force_mag / len(dataloader),
                        "VAL Total force vec loss": total_loss_force_vec / len(dataloader),
                        "VAL Total rot loss": total_loss_rot / len(dataloader),
                        "VAL Total trans loss": total_loss_trans / len(dataloader),
                        "VAL Total vertices loss": total_loss_vertices / len(dataloader)
                    }
        
                wandb.log(log_dict)

                # NOTE: logged under the "train/" wandb namespace to match the
                # original run's dashboards (unchanged from before this cleanup).
                log_wandb_visuals("train", epoch, T, batch, world_verts, verts_o_pred, pred_obj_force_magnitude, force_object,
                                   verts_l, verts_r, faces_l, faces_r,
                                   pred_left_force_magnitude, force_left, pred_right_force_magnitude, force_right,
                                   pred_contact_left, contact_left, pred_contact_right, contact_right, rgb_seq)

    return avg_loss

def load_checkpoint_if_exists(checkpoint_path, model, optimizer, device):
    """
    Resumes training from `checkpoint_path` if it exists (e.g. after a
    preemption on a shared cluster), otherwise starts fresh.
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} ...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = checkpoint.get("epoch", 0) + 1

        print(f"Checkpoint loaded (epoch {checkpoint['epoch']}), best val loss: {best_val_loss:.4f}")
        return model, optimizer, start_epoch, best_val_loss
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        return model, optimizer, 0, float("inf")



def main():
    # --- Distributed setup ---
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Default NCCL collective timeout is 10 min. Rank 0 alone writes
        # checkpoints (see below) to a network-mounted home directory, which
        # can legitimately take much longer than that under load -- other
        # ranks then block on the next all-reduce waiting for rank 0, and get
        # killed by the watchdog even though nothing is actually wrong.
        # Give collectives a lot more slack so slow I/O doesn't abort training.
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
    else:
        print("Running in non-distributed (single GPU) mode.")
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_distributed = False

    # --- Hyperparameters ---
    batch_size = 8
    seq_len = 1
    num_epochs = 1000
    learning_rate = 1e-4
    num_vertices_to_render = 200
    num_workers = 8
    feat_dim = 1024
    use_wandb = True

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    reg_loss_force = RegLossForce()
    vcb_loss = VCBLoss()
    reg_loss = RegLoss()
    smooth_loss = SmoothRegLoss()

    # Prepare MANO edges once
    hand_edges_left, hand_edges_right, hand_normals_left, hand_normals_right, = prepare_mano_edges()

    model = InteractionGNN(d_model=512, n_head=4, num_pose_blocks=4, num_force_blocks=4,
                 backbone_feature_dim=768, num_pose_iterations=3)

    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print(f"Using DistributedDataParallel on rank {local_rank}")
    else:
        print(f"Using single GPU on device {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    if use_wandb and ((not is_distributed) or (dist.get_rank() == 0)):
        wandb.init(project="arctic_distance_prediction", config={
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        })

    dataset = ArcticSequenceDataset(
        images_root=config.IMAGES_ROOT,
        segment_root=config.SEGMENT_ROOT,
        contacts_root=config.CONTACTS_ROOT,
        force_root=config.FORCE_ROOT,
        distances_root=config.DISTANCES_ROOT,
        processed_seqs_root=config.PROCESSED_SEQS_ROOT,
        mesh_root=config.MESH_ROOT,
        GT_mano_root=config.GT_MANO_ROOT,
        cameras=[0],
        sequence_length=seq_len,
        device=None
    )

    train_indices = []
    val_indices = []

    for i, sample in enumerate(dataset.samples):
        participant = sample['participant']
        if participant == 's05':
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_batches = group_by_object(train_indices, dataset)
    val_batches = group_by_object(val_indices, dataset)

    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0

    train_sampler = SameObjectBatchSampler(
        train_batches, 
        batch_size=batch_size, 
        shuffle=True,
        num_replicas=world_size,
        rank=rank
    )

    val_sampler = SameObjectBatchSampler(
        val_batches, 
        batch_size=batch_size, 
        shuffle=True,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=1)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=1)
    print(f"{len(train_loader)} train batches per epoch")

    # Resume from the last checkpoint if one exists (e.g. after a preemption).
    model, optimizer, start_epoch, best_val_loss = load_checkpoint_if_exists(
        config.LAST_CHECKPOINT_PATH, model, optimizer, device
    )

    for epoch in range(start_epoch, num_epochs):
        print('Epoch  ', epoch)
        
        # Train
        train_loss = train_one_epoch(use_wandb, model, train_loader, optimizer, device, reg_loss_force, vcb_loss, reg_loss, smooth_loss, is_distributed,
        hand_edges_left.to(device), hand_edges_right.to(device), hand_normals_left.to(device), hand_normals_right.to(device), num_vertices_to_render, epoch
        )
        
        val_loss = evaluate(use_wandb,
            model, val_loader,
            device, reg_loss_force, vcb_loss, reg_loss, smooth_loss, is_distributed, hand_edges_left.to(device), hand_edges_right.to(device), hand_normals_left.to(device), hand_normals_right.to(device), num_vertices_to_render, epoch, val=True,
        )

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if (not is_distributed) or (dist.get_rank() == 0):
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            checkpoint = {
                    "epoch": epoch,
                    "model_state": state_dict,
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss
                }

            if val_loss < best_val_loss:
                model.eval()
                best_val_loss = val_loss
                checkpoint["best_val_loss"] = best_val_loss
                torch.save(checkpoint, config.BEST_CHECKPOINT_PATH)
                print(f"Saved new best model with val_loss: {best_val_loss:.4f}")
            # Always save last checkpoint
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, config.LAST_CHECKPOINT_PATH)

if __name__ == '__main__':
    main()