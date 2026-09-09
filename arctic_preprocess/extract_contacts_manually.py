import numpy as np
import torch
import os
import trimesh
import pickle
from pathlib import Path
from joblib import Parallel, delayed
from config import MESH_ROOT, PROCESSED_SEQS_ROOT, IMAGES_ROOT, CONTACTS_ROOT, MANO_ROOT

def compute_contacts_with_penetration(vertices_obj, vertices_left, vertices_right,
                                      faces_obj, faces_left, faces_right,
                                      threshold):
    """
    CPU version of contact computation (distance + penetration) for a single sequence.
    """
    num_frames, num_obj, _ = vertices_obj.shape
    num_left = vertices_left.shape[1]
    num_right = vertices_right.shape[1]

    object_contacts = np.zeros((num_frames, num_obj, 1), dtype=np.uint8)
    left_contacts = np.zeros((num_frames, num_left, 1), dtype=np.uint8)
    right_contacts = np.zeros((num_frames, num_right, 1), dtype=np.uint8)

    for f in range(num_frames):
        obj_v = vertices_obj[f]
        left_v = vertices_left[f]
        right_v = vertices_right[f]

        # Penetration detection
        mesh_obj = trimesh.Trimesh(vertices=obj_v, faces=faces_obj, process=False)
        mesh_left = trimesh.Trimesh(vertices=left_v, faces=faces_left, process=False)
        mesh_right = trimesh.Trimesh(vertices=right_v, faces=faces_right, process=False)

        obj_inside_left = mesh_left.contains(obj_v)
        obj_inside_right = mesh_right.contains(obj_v)
        object_penetration_mask = obj_inside_left | obj_inside_right

        left_inside_obj = mesh_obj.contains(left_v)
        right_inside_obj = mesh_obj.contains(right_v)

        # Distance threshold detection (CPU)
        dist_left = np.linalg.norm(obj_v[:, None, :] - left_v[None, :, :], axis=-1)
        dist_right = np.linalg.norm(obj_v[:, None, :] - right_v[None, :, :], axis=-1)
        object_distance_mask = (dist_left.min(axis=1) < threshold) | (dist_right.min(axis=1) < threshold)

        dist_obj_left = np.linalg.norm(left_v[:, None, :] - obj_v[None, :, :], axis=-1)
        left_distance_mask = dist_obj_left.min(axis=1) < threshold

        dist_obj_right = np.linalg.norm(right_v[:, None, :] - obj_v[None, :, :], axis=-1)
        right_distance_mask = dist_obj_right.min(axis=1) < threshold

        # Combine penetration + distance
        object_contacts[f, :, 0] = np.logical_or(object_penetration_mask, object_distance_mask)
        left_contacts[f, :, 0] = np.logical_or(left_inside_obj, left_distance_mask)
        right_contacts[f, :, 0] = np.logical_or(right_inside_obj, right_distance_mask)

    return object_contacts, left_contacts, right_contacts

def process_sequence(seq_folder, faces_left, faces_right, threshold, output_folder):
    seq_name = seq_folder.name
    parts = seq_name.split("_")
    if len(parts) < 3:
        print(f"Skipping folder with unexpected name: {seq_name}")
        return

    object_name = parts[0]
    intent = "_".join(parts[1:-1])  # 'use_01'
    ID = parts[-1]  # 'retake'

    mesh_path = os.path.join(MESH_ROOT, object_name, "mesh.obj")
    mesh = trimesh.load(mesh_path, process=False)
    faces = mesh.faces

    seq_p = os.path.join(PROCESSED_SEQS_ROOT, seq_folder.parent.name,f"{seq_name}.npy")

    if not os.path.exists(seq_p):
        print(f"Vertex file does not exist: {seq_p}, skipping")
        return
    data = np.load(seq_p, allow_pickle=True).item()

    vertices = data['cam_coord']['verts.object'][:, 0, :, :]
    vertices_left = data['cam_coord']['verts.left'][:, 0, :, :]
    vertices_right = data['cam_coord']['verts.right'][:, 0, :, :]

    object_forces_all, left_forces_all, right_forces_all = compute_contacts_with_penetration(
        vertices, vertices_left, vertices_right,
        faces, faces_left, faces_right, threshold
    )

    out_sid_folder = Path(output_folder) / seq_folder.parent.name
    out_sid_folder.mkdir(parents=True, exist_ok=True)
    np.save(out_sid_folder / f"{object_name}_{intent}_{ID}_object.npy", object_forces_all)
    np.save(out_sid_folder / f"{object_name}_{intent}_{ID}_left.npy", left_forces_all)
    np.save(out_sid_folder / f"{object_name}_{intent}_{ID}_right.npy", right_forces_all)

    print(f"[{seq_folder.parent.name}] Processed sequence {seq_name}")


from pathlib import Path
from joblib import Parallel, delayed

def main_single_sequence(root_folder, output_folder, faces_left, faces_right, threshold=0.0025):
    root_folder = Path(root_folder)
    participant = "s10"
    seq_name = "phone_use_01_retake"

    seq_folder = root_folder / participant / seq_name
    if not seq_folder.exists():
        print(f"Sequence folder does not exist: {seq_folder}")
        return

    process_sequence(seq_folder, faces_left, faces_right, threshold, output_folder)
    print(f"Done processing {participant}/{seq_name}")

if __name__ == "__main__":
    root_folder = IMAGES_ROOT
    output_folder = CONTACTS_ROOT
    threshold = 0.0025

    mano_path = MANO_ROOT
    import pickle
    with open(Path(mano_path) / "MANO_LEFT.pkl", "rb") as f:
        faces_left = pickle.load(f, encoding="latin1")["f"]
    with open(Path(mano_path) / "MANO_RIGHT.pkl", "rb") as f:
        faces_right = pickle.load(f, encoding="latin1")["f"]

    main_single_sequence(root_folder, output_folder, faces_left, faces_right, threshold)
