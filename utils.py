import io
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from manopth.manolayer import ManoLayer

import config

# HACO_RELEASE is a third-party dependency cloned outside this repo; see config.py.
sys.path.append(config.HACO_RELEASE_ROOT)
from lib.core.config import cfg

def weighted_global_l2_loss(pred_force, gt_force, mask, contact_weight=10.0):
    """
    Calculates L2 loss globally, applying a high weight (contact_weight) 
    only to the contact regions.

    pred_force: [B, T, V, D]
    gt_force:   [B, T, V, D]
    mask:       [B, T, V, 1] (0/1 contact mask)
    contact_weight: Multiplier for errors in contact areas.
    """
    pred_force = torch.nan_to_num(pred_force, nan=0.0, posinf=1e6, neginf=-1e6)
    gt_force = torch.nan_to_num(gt_force, nan=0.0, posinf=1e6, neginf=-1e6)

    diff = pred_force - gt_force
    squared_diff = diff ** 2  # [B, T, V, D]

    # Sum over force dimensions (D=3 for force, D=1 for magnitude)
    element_wise_loss = torch.sum(squared_diff, dim=-1)  # [B, T, V]

    weights = torch.ones_like(mask)
    weights[mask > 0] = contact_weight
    weights = weights.squeeze(-1)  # [B, T, V]

    weighted_loss = weights * element_wise_loss

    loss = weighted_loss.mean()

    return loss


def to_device(device, *args):
    """
    Move all inputs to the given device and convert to float32 if they are floats.
    
    Args:
        device: torch.device
        *args: arbitrary number of tensors (or nested lists/tuples of tensors)
    
    Returns:
        Tensors moved to device and converted to float32.
    """
    def _convert(t):
        if isinstance(t, torch.Tensor):
            if t.dtype in [torch.float64, torch.float32]:
                return t.to(device=device, dtype=torch.float32)
            else:
                return t.to(device=device)
        elif isinstance(t, (list, tuple)):
            return type(t)(_convert(x) for x in t)
        else:
            return t
    return [_convert(x) for x in args]


def non_negativity_loss_diffusion(pred_force, surface_normals, contact_mask):
    """
    Penalizes force vectors that are not compressive (i.e., pointing outwards).

    Args:
        pred_force (torch.Tensor): Predicted force vectors, shape (B, T, V, 3).
        surface_normals (torch.Tensor): Outward-facing surface normals, shape (B, T, V, 3).
        contact_mask (torch.Tensor): Binary contact mask, shape (B, T, V, 1).

    Returns:
        torch.Tensor: The calculated non-negativity loss.
    """
    normals_norm = torch.norm(surface_normals, dim=-1, keepdim=True)
    normalized_normals = surface_normals / (normals_norm + 1e-8)

    # A positive dot product means the force is pointing outwards (tensile).
    dot_product = torch.sum(pred_force * normalized_normals, dim=-1, keepdim=True)

    masked_dot_product = (dot_product * contact_mask).clone()
    loss = torch.sum(torch.relu(masked_dot_product)).detach()

    num_contact_points = torch.sum(contact_mask)
    if num_contact_points == 0:
        return torch.tensor(0.0, device=pred_force.device, dtype=pred_force.dtype)

    return loss / num_contact_points

def compute_vertex_normals(verts, faces):
    """
    Compute vertex normals for a mesh.
    
    verts: [V, 3] tensor of vertex positions
    faces: [F, 3] tensor of vertex indices
    returns: [V, 3] tensor of vertex normals
    """
    v1 = verts[faces[:, 0]]
    v2 = verts[faces[:, 1]]
    v3 = verts[faces[:, 2]]
    face_normals = torch.cross(v2 - v1, v3 - v1, dim=1)  # [F, 3]

    vertex_normals = torch.zeros_like(verts)
    for i in range(3):
        idx = faces[:, i]
        vertex_normals.index_add_(0, idx, face_normals)

    vertex_normals = vertex_normals / (vertex_normals.norm(dim=1, keepdim=True) + 1e-8)

    return vertex_normals


def prepare_mano_edges(normalize=False):
    mano_layer_right = ManoLayer(flat_hand_mean=False, side='right', use_pca=False, mano_root=config.MANO_ROOT)
    mano_layer_left = ManoLayer(flat_hand_mean=False, side='left', use_pca=False, mano_root=config.MANO_ROOT)
    
    faces_r = mano_layer_right.th_faces
    faces_l = mano_layer_left.th_faces

    verts_r = mano_layer_right.th_v_template.squeeze(0)
    verts_l = mano_layer_left.th_v_template.squeeze(0)

    def faces_to_edges(faces):
        edges = torch.cat([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ], dim=0)
        edges = torch.unique(edges, dim=0)
        return edges

    edges_l = faces_to_edges(faces_l)
    edges_r = faces_to_edges(faces_r)

    normals_r = compute_vertex_normals(verts_r, faces_r)
    normals_l = compute_vertex_normals(verts_l, faces_l)

    return edges_l, edges_r, normals_l, normals_r


def knn_1nn(src, dst):
    """
    Find nearest neighbor indices in dst for each point in src.
    src: [B, N, 3]
    dst: [B, M, 3]
    Returns: [B, N] tensor of nearest neighbor indices in dst
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    src_exp = src.unsqueeze(2)   # [B, N, 1, 3]
    dst_exp = dst.unsqueeze(1)   # [B, 1, M, 3]

    dists = torch.norm(src_exp - dst_exp, dim=-1)  # [B, N, M]

    nn_idx = torch.argmin(dists, dim=-1)  # [B, N]
    return nn_idx


def geometric_consistency_loss_diffusion(
    force_left, force_right, force_obj,
    hand_verts_left, hand_verts_right, obj_verts,
    contact_mask_left, contact_mask_right, contact_mask_obj,
    eps=1e-8
):
    """
    Computes geometric consistency between hand and object forces at contact points.
    Stable against zero vectors in norm.
    """
    loss = torch.tensor(0.0, device=force_left.device)

    B, T, V_hand, _ = hand_verts_left.shape
    V_obj = obj_verts.shape[2]

    force_left_flat = force_left.reshape(B*T, V_hand, 3)
    force_right_flat = force_right.reshape(B*T, V_hand, 3)
    force_obj_flat = force_obj.reshape(B*T, V_obj, 3)

    hand_verts_left_flat = hand_verts_left.reshape(B*T, V_hand, 3)
    hand_verts_right_flat = hand_verts_right.reshape(B*T, V_hand, 3)
    obj_verts_flat = obj_verts.reshape(B*T, V_obj, 3)

    contact_mask_left_flat = contact_mask_left.reshape(B*T, V_hand, 1)
    contact_mask_right_flat = contact_mask_right.reshape(B*T, V_hand, 1)
    contact_mask_obj_flat = contact_mask_obj.reshape(B*T, V_obj, 1)

    contact_l_verts = hand_verts_left_flat * contact_mask_left_flat
    contact_r_verts = hand_verts_right_flat * contact_mask_right_flat

    def stable_norm(x):
        return torch.sqrt(torch.sum(x**2, dim=-1) + eps)

    # Left Hand - Object
    if torch.sum(contact_mask_left_flat) > 0 and torch.sum(contact_mask_obj_flat) > 0:
        nn_obj_idx = knn_1nn(contact_l_verts, obj_verts_flat)  # [B*T, V_hand]
        l_forces_at_contact = force_left_flat[contact_mask_left_flat.squeeze(-1).bool()]
        obj_forces_at_l_contact = force_obj_flat.gather(
            1, nn_obj_idx.unsqueeze(-1).repeat(1, 1, 3)
        ).clone()
        obj_forces_at_l_contact = obj_forces_at_l_contact[contact_mask_left_flat.squeeze(-1).bool()]
        loss += stable_norm(l_forces_at_contact + obj_forces_at_l_contact).mean()

    # Right Hand - Object
    if torch.sum(contact_mask_right_flat) > 0 and torch.sum(contact_mask_obj_flat) > 0:
        nn_obj_idx = knn_1nn(contact_r_verts, obj_verts_flat)  # [B*T, V_hand]
        r_forces_at_contact = force_right_flat[contact_mask_right_flat.squeeze(-1).bool()]
        obj_forces_at_r_contact = force_obj_flat.gather(
            1, nn_obj_idx.unsqueeze(-1).repeat(1, 1, 3)
        ).clone()
        obj_forces_at_r_contact = obj_forces_at_r_contact[contact_mask_right_flat.squeeze(-1).bool()]
        loss += stable_norm(r_forces_at_contact + obj_forces_at_r_contact).mean()

    return loss


def rotation_loss(R_pred, R_gt):
    """
    r_pred, r_gt: [B,T,3] axis-angle
    """
    B, T, _, _ = R_pred.shape

    # Geodesic distance
    RT = torch.matmul(R_pred.transpose(-1,-2), R_gt)
    trace = RT[...,0,0] + RT[...,1,1] + RT[...,2,2]  # [B,T]
    loss = torch.acos(torch.clamp((trace - 1)/2, -1+1e-7, 1-1e-7))
    return loss.mean()

def translation_loss(t_pred, t_gt):
    """
    t_pred, t_gt: [B,T,3]
    """
    return F.mse_loss(t_pred, t_gt)

def articulation_loss(arti_pred, arti_gt):
    """
    arti_pred, arti_gt: [B,T,1]
    """
    return F.mse_loss(arti_pred, arti_gt)

def scale_loss(s_pred, s_gt):
    return F.mse_loss(s_pred, s_gt)


def weighted_bce_loss(pred, target):
    """
    pred: (N,) or (N, num_vertices) raw logits or probabilities
    target: same shape, with binary ground truth {0,1}
    """
    pred = pred.reshape(-1)
    pred = pred.sigmoid()
    target = target.reshape(-1)

    N = target.numel()
    pos = target.sum()
    neg = N - pos + 1e-6  # avoid division by zero

    # class weights (Eq. 10 in the paper)
    w1 = N / (2.0 * pos.clamp(min=1))  # weight for positives
    w0 = N / (2.0 * neg)               # weight for negatives

    # BCE with manual weights (Eq. 9)
    loss = - (w1 * target * torch.log(pred.clamp(min=1e-8)) +
              w0 * (1 - target) * torch.log((1 - pred).clamp(min=1e-8)))
    return loss.mean()


def render_force_colored_mesh_torch(vertices, faces, forces, frame_indices, batch_idx=0):
    """
    Render selected frames of a mesh with vertices colored by force magnitudes.

    Args:
        vertices: (B, T, V, 3) torch tensor
        faces:    (B, T, F, 3) torch tensor
        forces:   (B, T, V, 3) torch tensor
        frame_indices: list of int, which frames to render
        batch_idx: int, which batch element to render (default: 0)

    Returns:
        dict mapping frame_index -> (front_img, back_img),
        where each img is a numpy array (H, W, 3).
    """
    results = {}

    for t in frame_indices:
        v = vertices[batch_idx, t]
        f = faces[batch_idx, t]
        fo = forces[batch_idx, t]

        # --- Force magnitude and normalization ---
        if fo.ndim == 2:  # case: vector forces (V,3)
            magnitudes = torch.norm(fo, dim=1)
        else:  # case: magnitudes already provided (V,)
            magnitudes = fo
        min_val, max_val = magnitudes.min(), magnitudes.max()
        normed = (magnitudes - min_val) / (max_val - min_val + 1e-8)

        cmap = cm.get_cmap("jet")
        colors = torch.from_numpy(cmap(normed.detach().cpu().numpy())[:, :4])  # RGBA

        verts_np = v.detach().cpu().numpy() * 1000
        faces_np = f.detach().cpu().numpy()
        face_colors = colors[f.cpu()].mean(dim=1).numpy()
        mesh_faces = verts_np[faces_np]

        def render_view(elev, azim):
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.axis("off")

            poly = Poly3DCollection(mesh_faces, facecolors=face_colors,
                                    linewidths=0.05, edgecolors="k", alpha=1.0)
            ax.add_collection3d(poly)
            ax.view_init(elev=elev, azim=azim)

            scale = verts_np.flatten()
            ax.auto_scale_xyz(scale, scale, scale)

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            img = np.array(Image.open(buf))
            buf.close()
            return img

        front_img = render_view(elev=45, azim=90)
        back_img = render_view(elev=45, azim=270)

        results[t] = (front_img, back_img)

    return results


def render_contact_hand(vertices, faces, contacts, frame_indices, batch_idx=0, threshold=0.5):
    """
    Render selected frames of a mesh with vertices colored by contact.
    Blue = no contact, Yellow = contact.

    Args:
        vertices: (B, T, V, 3) torch tensor
        faces:    (B, T, F, 3) torch tensor
        contacts: (B, T, V) torch tensor, values [0,1]
        frame_indices: list of int, which frames to render
        batch_idx: int, which batch element to render (default: 0)
        threshold: float, threshold to binarize contact (default=0.5)

    Returns:
        dict mapping frame_index -> (front_img, back_img),
        where each img is a numpy array (H, W, 3).
    """
    results = {}

    for t in frame_indices:
        v = vertices
        f = faces
        c = contacts[batch_idx, t]

        # --- Threshold contact to 0/1 ---
        binary_contact = (c > threshold).float()  # 1 = contact, 0 = no contact

        # No contact = blue (0,0,255), contact = yellow (255,255,0)
        colors = torch.zeros((v.shape[0], 3))
        colors[binary_contact == 0] = colors.new_tensor([0, 0, 1.0])
        colors[binary_contact == 1] = colors.new_tensor([1.0, 1.0, 0])

        verts_np = v.detach().cpu().numpy()
        faces_np = f.detach().cpu().numpy()
        face_colors = colors[f.cpu()].mean(dim=1).numpy()
        mesh_faces = verts_np[faces_np]

        def render_view(elev, azim):
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.axis("off")

            poly = Poly3DCollection(mesh_faces, facecolors=face_colors,
                                    linewidths=0.05, edgecolors="k", alpha=1.0)
            ax.add_collection3d(poly)
            ax.view_init(elev=elev, azim=azim)

            min_val = verts_np.min()
            max_val = verts_np.max()
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_zlim(min_val, max_val)

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            img = np.array(Image.open(buf))
            buf.close()
            return img

        front_img = render_view(elev=0, azim=90)
        back_img = render_view(elev=0, azim=270)

        results[t] = (front_img, back_img)

    return results


def render_force_hand(vertices, faces, forces, frame_indices, batch_idx=0):
    """
    Render selected frames of a mesh with vertices colored by force magnitudes.

    Args:
        vertices: (B, T, V, 3) torch tensor
        faces:    (B, T, F, 3) torch tensor
        forces:   (B, T, V, 3) torch tensor
        frame_indices: list of int, which frames to render
        batch_idx: int, which batch element to render (default: 0)

    Returns:
        dict mapping frame_index -> (front_img, back_img),
        where each img is a numpy array (H, W, 3).
    """
    results = {}

    for t in frame_indices:
        v = vertices
        f = faces
        fo = forces[batch_idx, t]

        # --- Force magnitude and normalization ---
        if fo.ndim == 2:  # case: vector forces (V,3)
            magnitudes = torch.norm(fo, dim=1)
        else:  # case: magnitudes already provided (V,)
            magnitudes = fo
        min_val, max_val = magnitudes.min(), magnitudes.max()
        normed = (magnitudes - min_val) / (max_val - min_val + 1e-8)

        cmap = cm.get_cmap("jet")
        colors = torch.from_numpy(cmap(normed.detach().cpu().numpy())[:, :4])  # RGBA

        verts_np = v.detach().cpu().numpy()
        faces_np = f.detach().cpu().numpy()
        face_colors = colors[f.cpu()].mean(dim=1).numpy()
        mesh_faces = verts_np[faces_np]

        def render_view(elev, azim):
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.axis("off")

            poly = Poly3DCollection(mesh_faces, facecolors=face_colors,
                                    linewidths=0.05, edgecolors="k", alpha=1.0)
            ax.add_collection3d(poly)
            ax.view_init(elev=elev, azim=azim)

            scale = verts_np.flatten()
            ax.auto_scale_xyz(scale, scale, scale)

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            img = np.array(Image.open(buf))
            buf.close()
            return img

        front_img = render_view(elev=0, azim=90)
        back_img = render_view(elev=0, azim=270)

        results[t] = (front_img, back_img)

    return results


#### NeurIPS paper losses


class VCBLoss(nn.Module):
    def __init__(self, beta=0.9999, v_type='mesh'):
        """
        Vertex-level class-Balanced Loss (VCB Loss)
        """
        super(VCBLoss, self).__init__()
        self.beta = beta
        self.v_type = v_type

        all_contact_samples = torch.from_numpy(np.load(cfg.MODEL.contact_data_path)).to(torch.int32)
        self.samples_per_cls = self.get_samples_per_cls(all_contact_samples, cfg.TRAIN.epoch)

        # BCE Loss without internal sigmoid
        self.bce_loss = nn.BCELoss(reduction='none')

    def get_samples_per_cls(self, all_contact_samples, total_epoch=10, total_sample_num=1000):  # total_sample_num is heuristic
        count_v_0 = (all_contact_samples == 0).sum(dim=0)
        count_v_1 = (all_contact_samples == 1).sum(dim=0)
        all_v_contact_counts = torch.stack([count_v_0, count_v_1], dim=1).float()

        count_0 = (all_contact_samples == 0).sum()
        count_1 = (all_contact_samples == 1).sum()
        all_contact_counts = torch.tensor([count_0, count_1]).float()

        denom_v = all_v_contact_counts.sum(dim=1, keepdim=True).clamp(min=1e-6)
        samples_per_cls_v = all_v_contact_counts / denom_v * total_sample_num

        denom = all_contact_counts.sum().clamp(min=1e-6)
        samples_per_cls = all_contact_counts / denom * total_sample_num

        # Weighted sum of VCB and CB loss: blend increasingly toward the
        # per-vertex distribution as training progresses.
        samples_per_cls_v_dict = {}
        v_weight_max = 0.3
        for epoch in range(total_epoch):
            v_weight = v_weight_max * (epoch / (total_epoch - 1))
            all_weight = 1.0 - v_weight
            samples_per_cls_v_epoch = v_weight * samples_per_cls_v.clone() + all_weight * samples_per_cls.clone() # Broadcast global [2] stats to [V, 2] for blending
            samples_per_cls_v_epoch = samples_per_cls_v_epoch.clamp(min=20.0)
            samples_per_cls_v_epoch = samples_per_cls_v_epoch / samples_per_cls_v_epoch.sum(dim=1, keepdim=True) * total_sample_num # re-normalize clamped values
            samples_per_cls_v_dict[epoch] = samples_per_cls_v_epoch

        if self.v_type == 'mesh':
            for epoch_key in samples_per_cls_v_dict:
                samples_per_cls_v_dict[epoch_key] = samples_per_cls_v_dict[epoch_key]
        else:
            raise NotImplementedError

        return samples_per_cls_v_dict

    def compute_class_weight(self, samples_per_cls, beta):
        """
        samples_per_cls: Tensor of shape [V, 2]  (class 0 count, class 1 count for each vertex)
        returns: class_weights of shape [V, 2]
        """
        effective_num = 1.0 - torch.pow(beta, samples_per_cls)      # [V, 2]
        class_weights = (1.0 - beta) / effective_num                # [V, 2]
        
        # Normalize per vertex (across classes) so sum over dim=1 is 1
        class_weights = class_weights / class_weights.sum(dim=1, keepdim=True)  # [V, 2]

        return class_weights  # [V, 2]
    
    def expand_class_weight(self, class_weights, pred, gt):
        w = class_weights[None, :, :] # class_weights: [num_verts, 2], w: [1, num_vertices, 2]
        g = gt.long()[..., None] # g: [batch, num_vertices, 1]
        class_weights = torch.gather(w.expand(gt.size(0), -1, -1), 2, g)[..., 0] # gt, pred: [batch_size, num_vertices], class_weights: [batch, num_vertices]
        class_weights_expanded = class_weights.to(pred.device) # gt, pred: [batch_size, num_vertices], self.class_weights: [2], class_weights: [batch_size, num_vertices]
        return class_weights_expanded

    def forward(self, pred, gt, epoch, valid=None): # pred: [batch, 778], gt: [batch, 778]
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            return torch.tensor(0.0, device=pred.device)
        gt = gt.to(pred.device).float()

        if valid is not None:
            if not valid.any().item():
                return torch.tensor(0.0, device=pred.device)
            pred, gt = pred[valid], gt[valid]

        pred = pred.sigmoid()
        pred = pred.clamp(1e-6, 1-1e-6)
        gt = gt.clamp(0.0, 1.0)

        if epoch not in self.samples_per_cls:
            epoch = max(self.samples_per_cls.keys())  # fallback to last epoch
        class_weights = self.compute_class_weight(self.samples_per_cls[epoch], self.beta)

        class_weights_expanded = self.expand_class_weight(class_weights.to(pred.device), pred, gt)
        vcb_loss = self.bce_loss(pred, gt) * class_weights_expanded  # [batch_size, num_vertices]

        # Penalize too confident positive predictions
        fp_reg_loss = ((gt == 0).float() * pred).mean()

        return vcb_loss.mean() + 0.5 * fp_reg_loss




class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred, obj_type=None, valid=None):
        batch_size = pred.shape[0]

        if obj_type==None:
            gt_means = np.load(cfg.MODEL.contact_means_path)
            gt_means = torch.tensor(gt_means).reshape(-1).repeat(batch_size, 1)
            gt_means = gt_means.to(pred.device).float()
        else:
            base_obj_mean_dir = config.OBJECT_MEANS_DIR
            obj_mean = os.path.join(base_obj_mean_dir, f"{obj_type}_mean.npy")
            gt_means = torch.tensor(np.load(obj_mean)).reshape(-1).repeat(batch_size, 1)
            gt_means = gt_means.to(pred.device).float()
        
        
        if valid is not None:
            if not valid.any().item():
                return torch.tensor(0, device=pred.device).float()
            pred, gt_means = pred[valid], gt_means[valid]

        pred = pred.sigmoid()

        return self.criterion(pred, gt_means)
    
class RegLossForce(nn.Module):
    def __init__(self):
        super(RegLossForce, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')
        json_path = config.SOFA_FORCE_LOG_MAGNITUDE_STATS_JSON
        with open(json_path, "r") as f:
            self.stats = json.load(f)

    def forward(self, pred, obj_type=None, hand=None, valid=None):
        batch_size = pred.shape[0]
        base_dir = config.FORCE_MEANS_DIR

        if hand=='left':
            obj_mean = os.path.join(base_dir, f"{obj_type}_left_mean.npy")
            gt_means = torch.tensor(np.load(obj_mean)).reshape(-1,3).repeat(batch_size, 1, 1)
            gt_means = gt_means.to(pred.device).float()
            suffix = 'left'
        elif hand=='right':
            obj_mean = os.path.join(base_dir, f"{obj_type}_right_mean.npy")
            gt_means = torch.tensor(np.load(obj_mean)).reshape(-1,3).repeat(batch_size, 1, 1)
            gt_means = gt_means.to(pred.device).float()
            suffix = 'right'
        else:
            obj_mean = os.path.join(base_dir, f"{obj_type}_object_mean.npy")
            gt_means = torch.tensor(np.load(obj_mean)).reshape(1,-1,3).repeat(batch_size, 1, 1)
            gt_means = gt_means.to(pred.device).float()
            suffix = 'object'
        
        magnitudes = torch.norm(gt_means, dim=2)  # [B, N_vertices]
        mag_log = torch.log1p(magnitudes)
        mean = self.stats[suffix]['mean']
        std = self.stats[suffix]['std']
        norm_magnitude = (mag_log - mean) / (std + 1e-8)  # standardized magnitude
        
        if valid is not None:
            if not valid.any().item():
                return torch.tensor(0, device=pred.device).float()
            pred, gt_means = pred[valid], gt_means[valid]

        pred = pred.sigmoid()

        return self.criterion(pred, norm_magnitude)


class SmoothRegLoss(nn.Module):
    def __init__(self):
        super(SmoothRegLoss, self).__init__()

    def build_adjacency_matrix(self, num_verts, faces):
        """ Constructs a sparse adjacency matrix for memory efficiency. """
        device = faces.device
        row_idx = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2],
                             faces[:, 1], faces[:, 2], faces[:, 0]])
        col_idx = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0],
                             faces[:, 0], faces[:, 1], faces[:, 2]])

        indices = torch.stack([row_idx, col_idx], dim=0)
        values = torch.ones(indices.shape[1], device=device)

        adjacency = torch.sparse_coo_tensor(indices, values,
                                            (num_verts, num_verts), device=device)
        
        # Add self-loops for improved stability
        self_loops = torch.eye(num_verts, device=device)
        adjacency += self_loops.to_sparse()

        return adjacency

    def forward(self, pred, faces):
        batch_size, num_verts = pred.shape
        pred_contact = pred.sigmoid()
        pred_non_contact = 1.0 - pred_contact

        faces = faces.to(pred.device).long()

        adjacency = self.build_adjacency_matrix(num_verts, faces)

        propagated_contact = torch.stack([
            torch.sparse.mm(adjacency, pred_contact[b].unsqueeze(-1)).squeeze(-1)
            for b in range(batch_size)
        ], dim=0)

        propagated_non_contact = torch.stack([
            torch.sparse.mm(adjacency, pred_non_contact[b].unsqueeze(-1)).squeeze(-1)
            for b in range(batch_size)
        ], dim=0)

        normalized_propagated_contact = propagated_contact / (
            propagated_contact.max(dim=1, keepdim=True)[0] + 1e-6
        )

        normalized_propagated_non_contact = propagated_non_contact / (
            propagated_non_contact.max(dim=1, keepdim=True)[0] + 1e-6
        )

        isolation_score_contact = torch.abs(pred_contact - normalized_propagated_contact)
        isolation_score_non_contact = torch.abs(pred_non_contact - normalized_propagated_non_contact)

        approx_total_contact_size = propagated_contact.sum(dim=1)
        approx_total_non_contact_size = propagated_non_contact.sum(dim=1)

        isolated_cluster_score = isolation_score_contact.sum(dim=1) + isolation_score_non_contact.sum(dim=1)
        norm_factor = (approx_total_contact_size + approx_total_non_contact_size) + 1e-3

        penalty_ratio = isolated_cluster_score / norm_factor
        loss_isolation = torch.log1p(penalty_ratio).mean()

        return loss_isolation
    

def extract_2d_bboxes(segmentation_mask_seq):
    """
    Extracts 2D bounding boxes (min_x, min_y, max_x, max_y) for each frame
    in a sequence of segmentation masks. max_x/max_y are exclusive (max + 1).
    Shared by train.py and the evaluation scripts.
    """
    num_frames, H, W = segmentation_mask_seq.shape
    bboxes = torch.zeros((num_frames, 4), dtype=torch.int32, device=segmentation_mask_seq.device)
    for i in range(num_frames):
        mask = segmentation_mask_seq[i]
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            pass
        else:
            min_y = torch.amin(coords[:, 0])
            max_y = torch.amax(coords[:, 0])
            min_x = torch.amin(coords[:, 1])
            max_x = torch.amax(coords[:, 1])
            bboxes[i, 0] = min_x
            bboxes[i, 1] = min_y
            bboxes[i, 2] = max_x + 1
            bboxes[i, 3] = max_y + 1
    return bboxes


def group_by_object(indices, dataset):
    """Groups sample indices by their object name, for SameObjectBatchSampler."""
    object_to_indices = defaultdict(list)
    for idx in indices:
        object_name = dataset.samples[idx]['object']
        object_to_indices[object_name].append(idx)
    return list(object_to_indices.values())


# --- Per-frame prediction figures (visualize_ARCTIC.py / visualize_H2O.py) --------------
#
# Independent of `render_force_hand` / `render_contact_hand` / `render_force_colored_mesh_torch`
# above, which are wired into the wandb training-time logging in train.py -- these build one
# combined multi-panel figure per frame on disk instead of separate wandb images.

def _face_colors(vertex_values, faces, cmap_name, vmin, vmax):
    """Average a per-vertex scalar over each face's vertices and map it through a colormap."""
    normed = np.clip((vertex_values - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    face_values = normed[faces].mean(axis=1)
    return plt.get_cmap(cmap_name)(face_values)


def _contact_weighted_view(vertices, faces, weights):
    """
    Pick a camera (elev, azim) that faces the mesh region with the highest
    `weights` (e.g. predicted contact probability or force magnitude), by
    pointing the view along the weight-averaged vertex normal. This keeps the
    "business side" of a hand/object -- its predicted contact surface, the
    palm for a grasping hand -- visible regardless of the mesh's actual
    orientation in camera space.
    """
    normals = compute_vertex_normals(
        torch.as_tensor(vertices, dtype=torch.float32),
        torch.as_tensor(faces, dtype=torch.long),
    ).numpy()
    weights = np.clip(weights, 0.0, None)
    if weights.sum() < 1e-8:
        direction = normals.mean(axis=0)
    else:
        direction = (normals * weights[:, None]).sum(axis=0) / weights.sum()
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
    azim = np.degrees(np.arctan2(direction[1], direction[0]))
    elev = np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    return elev, azim


def plot_scalar_mesh(ax, vertices, faces, values, cmap_name="Reds", vmin=0.0, vmax=1.0, view=None):
    """
    Draw one mesh into a 3D matplotlib axis, per-vertex-colored by `values`
    (e.g. contact probability or normalized force magnitude, both in
    [vmin, vmax]). Returns a ScalarMappable suitable for `fig.colorbar(...)`.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    values = np.asarray(values)

    poly = Poly3DCollection(
        vertices[faces],
        facecolors=_face_colors(values, faces, cmap_name, vmin, vmax),
        linewidths=0.1, edgecolors=(0.0, 0.0, 0.0, 0.15),
    )
    ax.add_collection3d(poly)

    center = vertices.mean(axis=0)
    radius = np.linalg.norm(vertices - center, axis=1).max() + 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(*(view if view is not None else _contact_weighted_view(vertices, faces, values)))

    return cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_name)


def add_force_quivers(ax, vertices, magnitudes, directions, top_k=40, color="black"):
    """Overlay arrows at the `top_k` highest-magnitude vertices, showing predicted force direction."""
    vertices = np.asarray(vertices)
    magnitudes = np.asarray(magnitudes)
    directions = np.asarray(directions)
    if len(vertices) == 0 or magnitudes.max() <= 0:
        return
    idx = np.argsort(magnitudes)[-min(top_k, len(vertices)):]
    scale = (np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)) + 1e-6) * 0.15
    arrow_len = (magnitudes[idx] / (magnitudes.max() + 1e-8)) * scale
    ax.quiver(
        vertices[idx, 0], vertices[idx, 1], vertices[idx, 2],
        directions[idx, 0] * arrow_len, directions[idx, 1] * arrow_len, directions[idx, 2] * arrow_len,
        color=color, linewidth=0.8, arrow_length_ratio=0.3,
    )


def _auto_brighten(image, low_percentile=1.0, high_percentile=99.0):
    """
    Percentile contrast-stretch for DISPLAY only -- some ARCTIC/H2O egocentric
    frames are genuinely dark straight off disk (that's what the model is
    actually trained/evaluated on, so it's left untouched everywhere else);
    this just makes the input-frame panel legible in the saved figure.
    """
    lo, hi = np.percentile(image, [low_percentile, high_percentile])
    if hi - lo < 1e-6:
        return image
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def render_prediction_figure(
    out_path,
    rgb_image,
    hand_vertices, hand_faces, hand_contact, hand_force_mag, hand_force_dir,
    object_vertices, object_faces, object_contact, object_force_mag, object_force_dir,
    title=None,
):
    """
    Save one composite figure for a single frame: the input RGB frame on the
    left, then a 2x2 grid of [hands, object] x [predicted contact, predicted
    force] on the right. `hand_*` arrays hold the concatenated left+right hand
    meshes (see `visualize_ARCTIC.py` / `visualize_H2O.py` for how they're
    built); force panels additionally overlay direction arrows on the
    top-magnitude vertices.
    """
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.1, 1, 1))

    ax_img = fig.add_subplot(grid[:, 0])
    ax_img.imshow(_auto_brighten(np.clip(rgb_image, 0.0, 1.0)))
    ax_img.set_title(title or "input frame")
    ax_img.axis("off")

    panels = (
        ("hand contact (pred)", grid[0, 1], hand_vertices, hand_faces, hand_contact, "Reds", None, None),
        ("object contact (pred)", grid[0, 2], object_vertices, object_faces, object_contact, "Reds", None, None),
        ("hand force (pred)", grid[1, 1], hand_vertices, hand_faces, hand_force_mag, "plasma", hand_force_mag, hand_force_dir),
        ("object force (pred)", grid[1, 2], object_vertices, object_faces, object_force_mag, "plasma", object_force_mag, object_force_dir),
    )
    for title_text, cell, verts, faces, values, cmap_name, force_mag, force_dir in panels:
        ax = fig.add_subplot(cell, projection="3d")
        mappable = plot_scalar_mesh(ax, verts, faces, values, cmap_name=cmap_name)
        if force_mag is not None:
            add_force_quivers(ax, verts, force_mag, force_dir)
        ax.set_title(title_text)
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)