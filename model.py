"""
InteractionGNN: joint hand-object pose and contact-force estimation model.

Pipeline:
  1. A ViT backbone extracts a dense 2D feature map from the egocentric RGB frame.
  2. Hand and object vertices are projected into the feature map and sampled
     (RoI-aligned for the object) to get per-vertex visual features.
  3. An iterative GNN + cross-attention stage ("pose interaction") refines the
     object's 6-DoF pose relative to the two hands.
  4. A second GNN + cross-attention stage ("force interaction"), conditioned on
     the final pose, predicts per-vertex contact probability and contact force.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from torchvision.ops import roi_align

from angular import r6d_to_rotation_matrix

DROPOUT_RATE = 0.0

def apply_articulation_batch(vertices, part_ids, part_id_to_move, radians, pivot, axis=None):
    """
    Fully vectorized batched articulation using provided pivots.

    Args:
        vertices: [B*T, N, 3]
        part_ids: [B*T, N]
        part_id_to_move: int
        radians: [B*T, 1] or [B*T]
        pivot: [B*T, 3]
        axis: [3] or None

    Returns:
        vertices_out: [B*T, N, 3]
    """
    if axis is None:
        axis = torch.tensor([0.0, 0.0, 1.0], device=vertices.device, dtype=vertices.dtype)
    axis = axis / torch.norm(axis)

    BxT, N, _ = vertices.shape
    device = vertices.device
    dtype = vertices.dtype

    mask = (part_ids == part_id_to_move)  # [B*T, N]

    pivot_expand = pivot.unsqueeze(1)  # [B*T, 1, 3]
    translated = vertices - pivot_expand  # [B*T, N, 3]

    # Rodrigues' formula: R = I + sin(r) K + (1-cos(r)) K^2
    K = torch.zeros((BxT, 3, 3), device=device, dtype=dtype)
    K[:, 0, 1] = -axis[2]
    K[:, 0, 2] = axis[1]
    K[:, 1, 0] = axis[2]
    K[:, 1, 2] = -axis[0]
    K[:, 2, 0] = -axis[1]
    K[:, 2, 1] = axis[0]

    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(BxT, -1, -1)  # [BxT, 3, 3]

    r = radians.view(-1, 1, 1)  # [B*T, 1, 1]
    R = I + torch.sin(r) * K + (1 - torch.cos(r)) * (K @ K)  # [BxT, 3, 3]

    rotated = torch.bmm(translated, R.transpose(1, 2))  # [BxT, N, 3]
    rotated += pivot_expand

    vertices_out = vertices.clone()
    mask_expand = mask.unsqueeze(-1)  # [B*T, N, 1]
    vertices_out = torch.where(mask_expand, rotated, vertices_out)

    return vertices_out


# --- 1. Visual backbone -----------------------------------------------------

class VisualBackbone(nn.Module):
    """
    ViT backbone that returns patch features as a 2D feature map.
    Output shape: [B, D, H/patch_size, W/patch_size]
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()
        if hasattr(self.vit, 'fc'):
            self.vit.fc = nn.Identity()

        self.patch_size = self.vit.patch_size if hasattr(self.vit, 'patch_size') else 16
        self.hidden_dim = self.vit.embed_dim  # e.g., 768

    def forward(self, x):
        """
        x: [B, 3, H, W]
        returns: [B, D, H/patch_size, W/patch_size]
        """
        features = self.vit.forward_features(x)  # [B, N_patches, D]

        # Drop the class token if the backbone prepends one.
        if features.shape[1] == (x.shape[2] // self.patch_size) * (x.shape[3] // self.patch_size) + 1:
            features = features[:, 1:, :]

        B, N, D = features.shape
        H_patch = W_patch = int(N ** 0.5)
        features_2d = features.transpose(1, 2).reshape(B, D, H_patch, W_patch)
        return features_2d

# --- 2. Vertex feature sampling ---------------------------------------------

def project_and_sample(vertices_cam, K, feature_map, H_orig, W_orig):
    """
    Projects 3D vertices (in camera space) to 2D and samples from a feature map.
    This version is DIFFERENTIABLE.

    Args:
        vertices_cam (Tensor): [B, N_verts, 3] vertices in camera coordinates.
        K (Tensor): [B, 3, 3] camera intrinsic matrix.
        feature_map (Tensor): [B, C, H_feat, W_feat] from the backbone.
    """
    B, N, _ = vertices_cam.shape
    device = vertices_cam.device

    # H_orig/W_orig: the original image's size. H_in/W_in: the size it was
    # resized to before the backbone (e.g. ViT-224) -- both are needed since
    # K and the projected pixel coordinates live in the *original* space.
    H_in, W_in = 224, 224

    fx = K[:, 0:1, 0:1]  # [B, 1, 1]
    fy = K[:, 1:2, 1:2]
    cx = K[:, 0:1, 2:3]
    cy = K[:, 1:2, 2:3]

    X, Y, Z = vertices_cam.split(1, dim=-1)  # [B, N, 1] each
    Z_safe = Z.clamp(min=1e-6)  # avoid divide-by-zero for points at Z=0

    u = (fx * X / Z_safe) + cx  # [B, N, 1] in [0, W_orig]
    v = (fy * Y / Z_safe) + cy  # [B, N, 1] in [0, H_orig]

    # Rescale from original image space to the backbone's input space before normalizing.
    scale_x = torch.tensor(W_in / W_orig, device=device, dtype=u.dtype).view(1, 1, 1)
    scale_y = torch.tensor(H_in / H_orig, device=device, dtype=v.dtype).view(1, 1, 1)
    u_scaled = u * scale_x
    v_scaled = v * scale_y

    # grid_sample expects normalized coordinates in [-1, 1].
    u_norm = (u_scaled / (W_in - 1)) * 2 - 1
    v_norm = (v_scaled / (H_in - 1)) * 2 - 1

    grid = torch.cat([u_norm, v_norm], dim=-1)  # [B, N, 2]
    grid = grid.unsqueeze(1)  # grid_sample wants [B, H_out, W_out, 2] -> [B, 1, N, 2]

    sampled_features = F.grid_sample(
        feature_map,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )  # [B, C, 1, N]

    return sampled_features.squeeze(2).permute(0, 2, 1)  # [B, N, C]


# --- 3. Graph-based Interaction Reasoning (GNN + Attention) ---

def batched_pyg_gnn(gnn_layer, h, edges):
    """
    Vectorized batched PyG GNN.

    Args:
        gnn_layer: a PyG GNN layer (e.g., GATConv) expecting (x, edge_index)
        h: [B, N, C] node features
        edges: [2, E] edge_index (same topology for every batch sample)
               dtype must be torch.long. Will be moved to h.device.

    Returns:
        h_out: [B, N, C_out]
    """
    B, N, C = h.shape
    device = h.device
    edges = edges.to(device)
    assert edges.dim() == 2 and edges.shape[0] == 2, "edges must be [2, E]"

    E = edges.shape[1]

    # Build one big edge_index for the B graphs concatenated into B*N nodes,
    # by offsetting each batch element's node indices by b*N.
    offsets = (torch.arange(B, device=device, dtype=edges.dtype) * N)  # [B]
    edges_expanded = edges.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)  # [2, E, B]
    edges_batch = edges_expanded.permute(2, 0, 1).reshape(2, B * E).contiguous()

    h_flat = h.reshape(B * N, C)
    edges_with_loops, _ = add_self_loops(edges_batch, num_nodes=B * N)
    h_out_flat = gnn_layer(h_flat, edges_with_loops)  # [B*N, C_out]

    C_out = h_out_flat.shape[-1]
    h_out = h_out_flat.view(B, N, C_out)

    return h_out


class InteractionBlock(nn.Module):
    """
    One block of GNN + Cross-Attention + FFN.
    """
    def __init__(self, d_model, n_head):
        super().__init__()

        # Intra-mesh GNN
        self.gat_l = GATConv(d_model, d_model, heads=n_head, concat=False)
        self.gat_r = GATConv(d_model, d_model, heads=n_head, concat=False)
        self.gat_o = GATConv(d_model, d_model, heads=n_head, concat=False)

        # Inter-mesh cross-attention
        self.mha_l = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.mha_r = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.mha_o = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        # Feed-forward
        self.ffn_l = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.ffn_r = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.ffn_o = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))

        # Norms
        self.norm1_l = nn.LayerNorm(d_model)
        self.norm2_l = nn.LayerNorm(d_model)
        self.norm3_l = nn.LayerNorm(d_model)

        self.norm1_r = nn.LayerNorm(d_model)
        self.norm2_r = nn.LayerNorm(d_model)
        self.norm3_r = nn.LayerNorm(d_model)

        self.norm1_o = nn.LayerNorm(d_model)
        self.norm2_o = nn.LayerNorm(d_model)
        self.norm3_o = nn.LayerNorm(d_model)

    def forward(self, h_l, h_r, h_o, edges_l, edges_r, edges_o):
        # --- Intra-mesh GNN ---
        h_l_gat = self.norm1_l(h_l + batched_pyg_gnn(self.gat_l, h_l, edges_l))
        h_r_gat = self.norm1_r(h_r + batched_pyg_gnn(self.gat_r, h_r, edges_r))
        h_o_gat = self.norm1_o(h_o + batched_pyg_gnn(self.gat_o, h_o, edges_o))

        # --- Inter-mesh cross-attention ---
        # Left hand attends to right hand + object
        context_lr = torch.cat([h_r_gat, h_o_gat], dim=1)
        h_l_ctx, _ = self.mha_l(query=h_l_gat, key=context_lr, value=context_lr)
        h_l_att = self.norm2_l(h_l_gat + h_l_ctx)

        # Right hand attends to left hand + object
        context_ro = torch.cat([h_l_gat, h_o_gat], dim=1)
        h_r_ctx, _ = self.mha_r(query=h_r_gat, key=context_ro, value=context_ro)
        h_r_att = self.norm2_r(h_r_gat + h_r_ctx)

        # Object attends to left hand + right hand
        context_oh = torch.cat([h_l_gat, h_r_gat], dim=1)
        h_o_ctx, _ = self.mha_o(query=h_o_gat, key=context_oh, value=context_oh)
        h_o_att = self.norm2_o(h_o_gat + h_o_ctx)

        # --- FFN update ---
        h_l_out = self.norm3_l(h_l_att + self.ffn_l(h_l_att))
        h_r_out = self.norm3_r(h_r_att + self.ffn_r(h_r_att))
        h_o_out = self.norm3_o(h_o_att + self.ffn_o(h_o_att))

        return h_l_out, h_r_out, h_o_out


# --- 4. Output Heads ---

class ForcePredictorHead(nn.Module):
    """
    Predicts contact and force from final vertex features.
    """
    def __init__(self, d_model):
        super().__init__()
        self.mlp_contact = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.mlp_force = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.mlp_force_vector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3)
        )

    def forward(self, h):
        # h: [B, N, d_model]
        p_contact = self.mlp_contact(h)         # [B, N, 1]
        force = self.mlp_force(h)               # [B, N, 1]
        force_vector = self.mlp_force_vector(h) # [B, N, 3]

        # Force is gated by contact probability.
        final_force = p_contact * force
        final_force_vector = p_contact * force_vector

        return p_contact, final_force, final_force_vector


# --- 5. The full model -------------------------------------------------------

class InteractionGNN(nn.Module):
    def __init__(self, d_model=1024, n_head=4, num_pose_blocks=4, num_force_blocks=2,
                 backbone_feature_dim=768, num_pose_iterations=3):

        super().__init__()

        self.d_model = d_model
        self.num_pose_iterations = num_pose_iterations

        # --- 1. Visual backbone ---
        self.backbone = VisualBackbone(model_name='vit_base_patch16_224', pretrained=True)

        # --- 2. Input embedders ---

        # Hand (cam geo + visual)
        self.vis_embed_hand = nn.Linear(backbone_feature_dim, d_model)
        self.geo_embed_hand = nn.Linear(3, d_model)
        self.input_mlp_hand = nn.Linear(d_model * 2, d_model)

        # Object pose input: visual (from RoI) + local geo + 2x hand-relative geo
        self.vis_embed_obj_pose = nn.Linear(backbone_feature_dim, d_model)
        self.geo_embed_obj_local = nn.Linear(3, d_model)
        self.geo_embed_obj_hand_rel_l = nn.Linear(3, d_model)
        self.geo_embed_obj_hand_rel_r = nn.Linear(3, d_model)
        self.input_mlp_obj_pose = nn.Linear(d_model * 4, d_model)

        # Object force input: visual (from full map) + cam geo
        self.vis_embed_obj_force = nn.Linear(backbone_feature_dim, d_model)
        self.geo_embed_obj_cam = nn.Linear(3, d_model)
        self.input_mlp_obj_force = nn.Linear(d_model * 2, d_model)

        # --- 3. Stage 1: pose interaction ---
        self.pose_interaction_blocks = nn.ModuleList([
            InteractionBlock(d_model, n_head) for _ in range(num_pose_blocks)
        ])

        # --- 4. Stage 1: pose decoder ---
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        fused_global_dim = d_model * 3  # hand_l + hand_r + obj

        self.pose_decoder_mlp = nn.Sequential(
            nn.Linear(fused_global_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        self.head_trans = nn.Linear(d_model, 3)
        self.head_rot = nn.Linear(d_model, 6)

        # --- 5. Stage 2: force interaction ---
        self.force_interaction_blocks = nn.ModuleList([
            InteractionBlock(d_model, n_head) for _ in range(num_force_blocks)
        ])

        # --- 6. Stage 2: force decoder ---
        self.head_hand = ForcePredictorHead(d_model)
        self.head_obj = ForcePredictorHead(d_model)

    def forward(self, x_stack,
                verts_l, verts_r,
                verts_l_center, verts_r_center,
                verts_articulated, scale,
                bbox_obj,  # object 2D bounding box [B, 4] (x1, y1, x2, y2)
                K, edges_l, edges_r, edges_o, H_orig, W_orig):

        B, N_l, _ = verts_l.shape
        N_obj, _ = verts_articulated.shape[1:3]
        device = x_stack.device

        # --- 1. 2D backbone & RoI feature extraction ---
        feat_map = self.backbone(x_stack)  # [B, C_feat, H_feat, W_feat]
        C_feat, H_feat, W_feat = feat_map.shape[1:]

        idx_tensor = torch.arange(B, device=device).float().view(-1, 1)
        roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)

        # ViT-224/16 -> H_feat x H_feat patch grid, so this recovers the
        # feature map's downsampling factor relative to the 224 input.
        spatial_scale = H_feat / 224.0

        feat_map_obj = roi_align(
            feat_map, roi_boxes_obj, output_size=(H_feat, W_feat),
            spatial_scale=spatial_scale,
            sampling_ratio=-1
        )  # [B, C_feat, H_feat, W_feat], RoI-aligned to the object bbox

        # --- 2. Initial feature prep (hand) ---
        feat_vis_l = self.vis_embed_hand(project_and_sample(verts_l, K, feat_map, H_orig, W_orig))
        feat_geo_l = self.geo_embed_hand(verts_l)
        h_l = F.gelu(self.input_mlp_hand(torch.cat([feat_vis_l, feat_geo_l], dim=-1)))

        feat_vis_r = self.vis_embed_hand(project_and_sample(verts_r, K, feat_map, H_orig, W_orig))
        feat_geo_r = self.geo_embed_hand(verts_r)
        h_r = F.gelu(self.input_mlp_hand(torch.cat([feat_vis_r, feat_geo_r], dim=-1)))

        # --- 3. Stage 1: iterative pose estimation ---
        R_matrix = torch.eye(3, device=device, dtype=x_stack.dtype).unsqueeze(0).expand(B, -1, -1)
        trans = torch.zeros((B, 3), device=device, dtype=x_stack.dtype)

        all_R = []
        all_T = []

        for i in range(self.num_pose_iterations):
            # Object vertices at the current pose estimate, used to sample fresh visual/geo features.
            verts_o_cam = torch.bmm(
                verts_articulated, R_matrix.transpose(1, 2)
            ) * scale.unsqueeze(-1).unsqueeze(-1) + trans.unsqueeze(1)

            feat_vis_o_pose = self.vis_embed_obj_pose(
                project_and_sample(verts_o_cam, K, feat_map_obj, H_orig, W_orig)
            )
            feat_geo_o_local = self.geo_embed_obj_local(verts_articulated)

            verts_l_center_expand = verts_l_center.unsqueeze(1).expand(-1, N_obj, -1)
            verts_r_center_expand = verts_r_center.unsqueeze(1).expand(-1, N_obj, -1)
            verts_o_hand_rel_l = verts_o_cam - verts_l_center_expand
            verts_o_hand_rel_r = verts_o_cam - verts_r_center_expand
            feat_geo_o_hand_rel_l = self.geo_embed_obj_hand_rel_l(verts_o_hand_rel_l)
            feat_geo_o_hand_rel_r = self.geo_embed_obj_hand_rel_r(verts_o_hand_rel_r)

            h_o_init = F.gelu(self.input_mlp_obj_pose(
                torch.cat([feat_vis_o_pose, feat_geo_o_local,
                           feat_geo_o_hand_rel_l, feat_geo_o_hand_rel_r], dim=-1)
            ))

            # Hand features stay fixed across pose iterations; only the object side updates.
            h_l_pose, h_r_pose, h_o_pose = h_l, h_r, h_o_init
            for block in self.pose_interaction_blocks:
                h_l_pose, h_r_pose, h_o_pose = block(
                    h_l_pose, h_r_pose, h_o_pose, edges_l.t(), edges_r.t(), edges_o
                )

            h_l_global = self.global_pool(h_l_pose.transpose(1, 2)).squeeze(-1)
            h_r_global = self.global_pool(h_r_pose.transpose(1, 2)).squeeze(-1)
            h_o_global = self.global_pool(h_o_pose.transpose(1, 2)).squeeze(-1)

            fused_global = torch.cat([h_l_global, h_r_global, h_o_global], dim=-1)
            pose_feat = self.pose_decoder_mlp(fused_global)

            delta_trans = self.head_trans(pose_feat)      # [B, 3]
            delta_rot_6d = self.head_rot(pose_feat)        # [B, 6]
            delta_R_matrix = r6d_to_rotation_matrix(delta_rot_6d)  # [B, 3, 3]

            R_matrix = torch.bmm(delta_R_matrix, R_matrix)  # R_new = delta_R @ R_prev
            trans = trans + delta_trans

            all_R.append(R_matrix.unsqueeze(1))
            all_T.append(trans.unsqueeze(1))

        T_final = trans
        R_final = R_matrix
        verts_o_pred = verts_o_cam  # pose from the final iteration

        # --- 4. Stage 2: force prediction ---
        # Unlike the pose stage, this samples the full feature map (not the RoI-cropped one).
        feat_vis_o_pred = self.vis_embed_obj_force(
            project_and_sample(verts_o_pred, K, feat_map, H_orig, W_orig)
        )
        feat_geo_o_pred = self.geo_embed_obj_cam(verts_o_pred)
        h_o_force_init = F.gelu(self.input_mlp_obj_force(
            torch.cat([feat_vis_o_pred, feat_geo_o_pred], dim=-1)
        ))

        # Re-use the pose-aware hand features from the final pose iteration.
        h_l_force, h_r_force, h_o_force = h_l_pose, h_r_pose, h_o_force_init
        for block in self.force_interaction_blocks:
            h_l_force, h_r_force, h_o_force = block(
                h_l_force, h_r_force, h_o_force, edges_l.t(), edges_r.t(), edges_o
            )

        contact_l, force_l, force_l_vector = self.head_hand(h_l_force)
        contact_r, force_r, force_r_vector = self.head_hand(h_r_force)
        contact_o, force_o, force_o_vector = self.head_obj(h_o_force)

        return (
            # Force/contact
            contact_l, contact_r, contact_o,
            force_l, force_r, force_o,
            force_l_vector, force_r_vector, force_o_vector,

            # Iterative poses (for training loss)
            torch.cat(all_T, dim=1),  # [B, num_iters, 3]
            torch.cat(all_R, dim=1),  # [B, num_iters, 3, 3]

            # Final pose (for inference / force stage)
            T_final,       # [B, 3]
            R_final,       # [B, 3, 3]
            verts_o_pred   # [B, N_obj, 3]
        )
