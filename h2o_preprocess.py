#!/usr/bin/env python3
"""Unified, in-process preprocessing pipeline for the H2O dataset.

This script performs every training-data preparation step without launching the
legacy helper scripts: RGB resizing, world/camera vertex extraction, object
rotation/translation/scale extraction, contact labels, and 224px masks.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
import trimesh
from PIL import Image

from config import DATA_ROOT_H2O, H2O_CONTACTS_ROOT, MANO_ROOT


OBJECT_NAMES = {
    1: "book", 2: "espresso", 3: "lotion", 4: "spray",
    5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino",
}
OBJ_POSE_LENGTH = 17
HAND_MANO_LENGTH = 124
DEFAULT_DATA_ROOT = Path(os.environ.get("H2O_DATA_ROOT", DATA_ROOT_H2O))
DEFAULT_CONTACT_ROOT = Path(os.environ.get("H2O_CONTACT_ROOT", H2O_CONTACTS_ROOT))


def iter_camera_dirs(data_root: Path):
    """Yield every H2O camera folder beneath every available participant."""
    for subject_dir in sorted(data_root.glob("subject*_ego")):
        if not subject_dir.is_dir():
            continue
        for path in sorted(subject_dir.rglob("cam*")):
            if path.is_dir() and path.name.startswith("cam") and path.name[3:].isdigit():
                yield path


def load_values(path: Path, expected_size: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        values = np.loadtxt(path, dtype=np.float32)
        values = np.atleast_1d(values).reshape(-1)
    except (OSError, ValueError) as error:
        print(f"[warning] Cannot read {path}: {error}")
        return None
    if values.size != expected_size:
        print(f"[warning] {path}: expected {expected_size} values, found {values.size}")
        return None
    return values


def first_scalar(path: Path) -> float | None:
    """Read the first whitespace-separated value from a text file."""
    try:
        with open(path) as handle:
            token = handle.read().split(None, 1)[0]
        return float(token)
    except (OSError, IndexError, ValueError) as error:
        print(f"[warning] Cannot read {path}: {error}")
        return None


def synchronized_frames(camera_dir: Path) -> list[int]:
    required = ("rgb", "cam_pose", "obj_pose_rt", "hand_pose_mano")
    frame_sets = []
    for folder in required:
        suffix = ".png" if folder == "rgb" else ".txt"
        paths = (camera_dir / folder).glob(f"*{suffix}")
        frame_sets.append({int(path.stem) for path in paths if path.stem.isdigit()})
    return sorted(set.intersection(*frame_sets)) if frame_sets else []


def load_mano_models(mano_model_root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vertices] Using {device}")
    left = smplx.create(model_path=str(mano_model_root.parent), model_type="mano", is_rhand=False,
                        use_pca=False, ncomps=45, flat_hand_mean=True).to(device)
    right = smplx.create(model_path=str(mano_model_root.parent), model_type="mano", is_rhand=True,
                         use_pca=False, ncomps=45, flat_hand_mean=True).to(device)
    return device, left, right


def mano_vertices(model, values: np.ndarray, offset: int, device: torch.device) -> tuple[np.ndarray, bool]:
    """Run the MANO forward pass for one hand, returning (vertices, valid).

    ``valid`` is False when this frame has no tracked pose for the hand; the
    vertices are then a zero placeholder that must NOT be run through any
    further transform (the reference leaves the world-frame output as a
    literal zero array in that case rather than transforming it).
    """
    valid = values[offset] == 1.0
    if not valid:
        return np.zeros((778, 3), dtype=np.float32), False
    translation = torch.as_tensor(values[offset + 1:offset + 4], device=device).unsqueeze(0)
    pose = torch.as_tensor(values[offset + 4:offset + 52], device=device).unsqueeze(0)
    shape = torch.as_tensor(values[offset + 52:offset + 62], device=device).unsqueeze(0)
    with torch.no_grad():
        output = model(betas=shape, hand_pose=pose[:, 3:], global_orient=pose[:, :3], transl=translation)
    return output.vertices[0].detach().cpu().numpy(), True


def to_world_frame(vertices: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """Transform camera-space vertices to world space with a 4x4 pose matrix."""
    homogeneous = np.c_[vertices, np.ones(len(vertices), dtype=np.float32)]
    return (camera_to_world @ homogeneous.T).T[:, :3]


def save_meshes(camera_dir: Path, suffix: str, vertices: dict[str, np.ndarray], faces: dict[str, np.ndarray]) -> None:
    for part, output_name in (("object", f"object_frame0_new{suffix}.obj"),
                              ("left_hand", f"left_hand_frame0_new{suffix}.obj"),
                              ("right_hand", f"right_hand_frame0_new{suffix}.obj")):
        trimesh.Trimesh(vertices=vertices[part][0], faces=faces[part], process=False).export(camera_dir / output_name)


def extract_vertices(data_root: Path, mano_model_root: Path, overwrite: bool) -> None:
    device, mano_left, mano_right = load_mano_models(mano_model_root)
    hand_faces = {"left_hand": np.asarray(mano_left.faces), "right_hand": np.asarray(mano_right.faces)}
    for camera_dir in iter_camera_dirs(data_root):
        world_output = camera_dir / "vertices_new.npy"
        camera_output = camera_dir / "vertices_new_cam.npy"
        if not overwrite and world_output.exists() and camera_output.exists():
            continue
        frames = synchronized_frames(camera_dir)
        if not frames:
            continue
        first_object = load_values(camera_dir / "obj_pose_rt" / f"{frames[0]:06d}.txt", OBJ_POSE_LENGTH)
        if first_object is None or int(first_object[0]) not in OBJECT_NAMES:
            continue
        object_name = OBJECT_NAMES[int(first_object[0])]
        mesh_path = data_root / "object" / object_name / f"{object_name}.obj"
        if not mesh_path.is_file():
            print(f"[vertices] Missing CAD model: {mesh_path}")
            continue
        mesh = trimesh.load(mesh_path, process=False)
        template = np.asarray(mesh.vertices, dtype=np.float32) @ np.diag([1.0, -1.0, -1.0])
        object_faces = np.asarray(mesh.faces)
        world = {key: [] for key in ("object", "left_hand", "right_hand")}
        camera = {key: [] for key in ("object", "left_hand", "right_hand")}
        for frame in frames:
            token = f"{frame:06d}.txt"
            camera_to_world = load_values(camera_dir / "cam_pose" / token, 16)
            object_pose = load_values(camera_dir / "obj_pose_rt" / token, OBJ_POSE_LENGTH)
            hands = load_values(camera_dir / "hand_pose_mano" / token, HAND_MANO_LENGTH)
            if camera_to_world is None or object_pose is None or hands is None:
                continue
            camera_to_world = camera_to_world.reshape(4, 4)
            object_to_camera = object_pose[1:].reshape(4, 4)
            # Combine into a single object-to-world matrix before applying it
            # (rather than chaining two matmuls against the vertices) so the
            # floating-point rounding matches the reference bit-for-bit --
            # matrix multiplication is associative in exact arithmetic but not
            # in float64, and the difference is large enough to occasionally
            # flip a boundary-case trimesh.contains() contact label.
            object_to_world = camera_to_world @ object_to_camera
            homogeneous = np.c_[template, np.ones(len(template), dtype=np.float32)]
            object_camera = (object_to_camera @ homogeneous.T).T[:, :3]
            object_world = (object_to_world @ homogeneous.T).T[:, :3]
            left_camera, left_valid = mano_vertices(mano_left, hands, 0, device)
            right_camera, right_valid = mano_vertices(mano_right, hands, 62, device)
            camera["object"].append(object_camera)
            camera["left_hand"].append(left_camera)
            camera["right_hand"].append(right_camera)
            world["object"].append(object_world)
            world["left_hand"].append(to_world_frame(left_camera, camera_to_world) if left_valid else left_camera)
            world["right_hand"].append(to_world_frame(right_camera, camera_to_world) if right_valid else right_camera)
        if not world["object"]:
            continue
        world = {key: np.stack(value) for key, value in world.items()}
        camera = {key: np.stack(value) for key, value in camera.items()}
        np.save(world_output, world)
        np.save(camera_output, camera)
        faces = {"object": object_faces, **hand_faces}
        save_meshes(camera_dir, "", world, faces)
        save_meshes(camera_dir, "_cam", camera, faces)
        print(f"[vertices] {camera_dir} ({len(world['object'])} synchronized frames)")


def resize_images(data_root: Path, overwrite: bool) -> None:
    for camera_dir in iter_camera_dirs(data_root):
        source = camera_dir / "rgb"
        destination = camera_dir / "rgb224"
        if not source.is_dir():
            continue
        destination.mkdir(exist_ok=True)
        for image_path in sorted(source.glob("*.png")):
            if not image_path.stem.isdigit():
                continue
            output = destination / image_path.name
            if output.exists() and not overwrite:
                continue
            with Image.open(image_path) as image:
                image.resize((224, 224), Image.Resampling.LANCZOS).save(output)


def similarity_transform(source: np.ndarray, target: np.ndarray):
    source_centered = source - source.mean(0)
    target_centered = target - target.mean(0)
    u, singular_values, vt = np.linalg.svd(source_centered.T @ target_centered / len(source))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    scale = singular_values.sum() / (np.square(source_centered).sum() / len(source))
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return rotation, translation, scale


def extract_transforms(data_root: Path, overwrite: bool) -> None:
    output_root = data_root / "extracted_rot_trans_scale"
    for camera_dir in iter_camera_dirs(data_root):
        vertices_path = camera_dir / "vertices_new_cam.npy"
        if not vertices_path.is_file():
            continue
        relative = camera_dir.relative_to(data_root)
        destination = output_root / relative
        result = destination / "rot_local_cam_normalized.npy"
        if result.exists() and not overwrite:
            continue
        pose_dir = camera_dir / "obj_pose"
        pose_files = sorted(pose_dir.glob("*.txt"))
        if not pose_files:
            continue
        # Note: "obj_pose" files hold a class id followed by 21 object
        # keypoints (64 floats total), NOT the 17-float class+4x4-matrix
        # layout used by "obj_pose_rt" -- only the leading class id is used
        # here, so it is read without enforcing an expected length.
        object_id = first_scalar(pose_files[0])
        if object_id is None or int(object_id) not in OBJECT_NAMES:
            continue
        object_name = OBJECT_NAMES[int(object_id)]
        mesh_path = data_root / "object" / object_name / f"{object_name}.obj"
        if not mesh_path.is_file():
            continue
        template = np.asarray(trimesh.load(mesh_path, process=False).vertices)
        template = template - template.mean(0, keepdims=True)
        template /= np.abs(template).max()
        targets = np.load(vertices_path, allow_pickle=True).item()["object"]
        rotations, translations, scales = [], [], []
        for target in targets:
            if len(target) != len(template):
                continue
            rotation, translation, scale = similarity_transform(template, target)
            rotations.append(rotation); translations.append(translation); scales.append(scale)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "rot_local_cam_normalized.npy", np.asarray(rotations))
        np.save(destination / "trans_local_cam_normalized.npy", np.asarray(translations))
        np.save(destination / "scale_local_cam_normalized.npy", np.asarray(scales))


def contact_labels(object_vertices, left_vertices, right_vertices, object_faces, left_faces, right_faces, threshold):
    outputs = [np.zeros((len(object_vertices), vertices.shape[1], 1), dtype=np.uint8)
               for vertices in (object_vertices, left_vertices, right_vertices)]
    for index, (obj, left, right) in enumerate(zip(object_vertices, left_vertices, right_vertices)):
        object_mesh = trimesh.Trimesh(vertices=obj, faces=object_faces, process=False)
        left_mesh = trimesh.Trimesh(vertices=left, faces=left_faces, process=False)
        right_mesh = trimesh.Trimesh(vertices=right, faces=right_faces, process=False)
        object_distance = (np.linalg.norm(obj[:, None] - left[None], axis=-1).min(1) < threshold) | (np.linalg.norm(obj[:, None] - right[None], axis=-1).min(1) < threshold)
        left_distance = np.linalg.norm(left[:, None] - obj[None], axis=-1).min(1) < threshold
        right_distance = np.linalg.norm(right[:, None] - obj[None], axis=-1).min(1) < threshold
        outputs[0][index, :, 0] = object_distance | left_mesh.contains(obj) | right_mesh.contains(obj)
        outputs[1][index, :, 0] = left_distance | object_mesh.contains(left)
        outputs[2][index, :, 0] = right_distance | object_mesh.contains(right)
    return outputs


def extract_contacts(data_root: Path, output_root: Path, threshold: float, overwrite: bool) -> None:
    with open(Path(MANO_ROOT) / "MANO_LEFT.pkl", "rb") as handle:
        left_faces = pickle.load(handle, encoding="latin1")["f"]
    with open(Path(MANO_ROOT) / "MANO_RIGHT.pkl", "rb") as handle:
        right_faces = pickle.load(handle, encoding="latin1")["f"]
    for vertices_path in data_root.rglob("vertices_new.npy"):
        camera_dir = vertices_path.parent
        relative = camera_dir.relative_to(data_root)
        destination = output_root / relative
        if (destination / "object_contacts.npy").exists() and not overwrite:
            continue
        mesh_path = camera_dir / "object_frame0_new.obj"
        if not mesh_path.is_file():
            continue
        data = np.load(vertices_path, allow_pickle=True).item()
        contacts = contact_labels(data["object"], data["left_hand"], data["right_hand"],
                                  np.asarray(trimesh.load(mesh_path, process=False).faces), left_faces, right_faces, threshold)
        destination.mkdir(parents=True, exist_ok=True)
        for name, labels in zip(("object_contacts.npy", "left_contacts.npy", "right_contacts.npy"), contacts):
            np.save(destination / name, labels)


def mesh_mask(vertices: np.ndarray, faces: np.ndarray, intrinsic: np.ndarray, height: int, width: int) -> np.ndarray:
    # The epsilon is added to depth itself (not just clamped as a divisor
    # floor) before the >0 validity check, matching the reference pixel-for-
    # pixel at grazing incidence (a vertex with depth in (-1e-8, 1e-8]).
    depth = vertices[:, 2] + 1e-8
    valid = depth > 0
    uv = np.zeros((len(vertices), 2), dtype=np.int32)
    uv[:, 0] = (intrinsic[0, 0] * vertices[:, 0] / depth + intrinsic[0, 2]).astype(np.int32)
    uv[:, 1] = (intrinsic[1, 1] * vertices[:, 1] / depth + intrinsic[1, 2]).astype(np.int32)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    index_map = np.full(len(vertices), -1)
    index_map[valid] = np.arange(valid.sum())
    valid_faces = faces[np.all(valid[faces], axis=1)]
    mask = np.zeros((height, width), dtype=np.uint8)
    for triangle in uv[valid][index_map[valid_faces]]:
        cv2.fillConvexPoly(mask, triangle, 1)
    return mask


def make_masks(data_root: Path, overwrite: bool) -> None:
    for camera_dir in iter_camera_dirs(data_root):
        vertices_path = camera_dir / "vertices_new_cam.npy"
        intrinsics_path = camera_dir / "cam_intrinsics.txt"
        object_mesh_path = camera_dir / "object_frame0_new_cam.obj"
        left_mesh_path = camera_dir / "left_hand_frame0_new_cam.obj"
        right_mesh_path = camera_dir / "right_hand_frame0_new_cam.obj"
        if not all(path.is_file() for path in (vertices_path, intrinsics_path, object_mesh_path, left_mesh_path, right_mesh_path)):
            continue
        values = np.loadtxt(intrinsics_path).reshape(-1)
        if len(values) < 6:
            continue
        fx, fy, cx, cy, width, height = values[:6]
        # float64 (not float32) to match the reference's precision -- the
        # intrinsics are large enough (fx, fy ~ 600-700) that float32
        # rounding can shift a projected pixel coordinate by a fraction of a
        # pixel, occasionally flipping which pixel an int32 cast lands on.
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        data = np.load(vertices_path, allow_pickle=True).item()
        meshes = {
            "object": np.asarray(trimesh.load(object_mesh_path, process=False).faces),
            "left_hand": np.asarray(trimesh.load(left_mesh_path, process=False).faces),
            "right_hand": np.asarray(trimesh.load(right_mesh_path, process=False).faces),
        }
        frame_files = sorted((camera_dir / "rgb224").glob("*.png"))
        frame_ids = [int(path.stem) for path in frame_files if path.stem.isdigit()] or list(range(len(data["object"])))
        destinations = {part: camera_dir / f"{part.replace('_hand', '')}_mask224" for part in meshes}
        for destination in destinations.values():
            destination.mkdir(exist_ok=True)
        for frame_id in frame_ids:
            if frame_id >= len(data["object"]):
                continue
            for part, faces in meshes.items():
                output = destinations[part] / f"{frame_id:06d}.png"
                if output.exists() and not overwrite:
                    continue
                mask = mesh_mask(data[part][frame_id], faces, intrinsic, int(height), int(width))
                mask = cv2.medianBlur(mask, 5)
                mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(output), (mask * 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--contacts-root", type=Path, default=DEFAULT_CONTACT_ROOT)
    parser.add_argument("--mano-root", type=Path, default=Path(MANO_ROOT))
    parser.add_argument("--stages", nargs="+", choices=("resize", "vertices", "transforms", "contacts", "segmentation"), default=("resize", "vertices", "transforms", "contacts", "segmentation"))
    parser.add_argument("--threshold", type=float, default=0.0025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"H2O data root not found: {args.data_root}")
    if "resize" in args.stages:
        resize_images(args.data_root, args.overwrite)
    if "vertices" in args.stages:
        extract_vertices(args.data_root, args.mano_root, args.overwrite)
    if "transforms" in args.stages:
        extract_transforms(args.data_root, args.overwrite)
    if "contacts" in args.stages:
        extract_contacts(args.data_root, args.contacts_root, args.threshold, args.overwrite)
    if "segmentation" in args.stages:
        make_masks(args.data_root, args.overwrite)


if __name__ == "__main__":
    main()
