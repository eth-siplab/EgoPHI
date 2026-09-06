#!/usr/bin/env python3
"""End-to-end preprocessing for every available ARCTIC participant.

The script produces 224px ego images, contact labels, ARCTIC object metadata
(rotation/translation/scale of the articulated object template in the world
frame, plus 3D bounding boxes), and rendered segmentation masks.  Paths
default to :mod:`config` and can be overridden at the command line so this
file is portable across machines.

Force simulation/estimation is out of scope here and handled elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

from config import (
    ARCTIC_ROOT, CONTACTS_ROOT, IMAGES_ROOT, MANO_ROOT, MESH_ROOT,
    PROCESSED_SEQS_ROOT, ROT_TRANS_ARTI_ROOT, SEGMENT_ROOT,
)

HERE = Path(__file__).resolve().parent

# transform_bbox3d_to_crop() in extract_rot_trans_arti.py scales the ego-view
# (view_idx 0) 2D bbox corners by this factor; the loose 2D bbox it also takes
# is only used for the other camera views, which this pipeline never renders.
EGO_BBOX_SCALE = 0.3


def _subjects(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("s"))


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return R, t, s such that target ~= s * (source @ R.T) + t."""
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


def _boundary_vertices(faces: np.ndarray, part_ids: np.ndarray) -> np.ndarray:
    edges: dict[tuple[int, int], set[bool]] = {}
    for face in faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(first), int(second))))
            edges.setdefault(key, set()).update((bool(part_ids[first]), bool(part_ids[second])))
    boundary = np.zeros(len(part_ids), dtype=bool)
    for (first, second), labels in edges.items():
        if len(labels) > 1:
            boundary[[first, second]] = True
    return boundary


def _articulate(vertices: np.ndarray, faces: np.ndarray, parts_file: Path, radians: float) -> np.ndarray:
    """Rotate the moving part (part id 1) of an ARCTIC object template by `radians`."""
    if not parts_file.exists():
        return vertices
    part_ids = np.asarray(json.loads(parts_file.read_text()))
    moving = part_ids == 1
    if not moving.any():
        return vertices
    boundary = _boundary_vertices(faces, moving)
    if not boundary.any():
        return vertices
    pivot = vertices[boundary].mean(0)
    # Fixed hinge axis used throughout the ARCTIC object templates.
    axis = np.array([0.0, 0.0, 1.0])
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + np.sin(radians) * cross + (1.0 - np.cos(radians)) * (cross @ cross)
    result = vertices.copy()
    result[moving] = (vertices[moving] - pivot) @ rotation.T + pivot
    return result


def _load_mesh(mesh_root: Path, object_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    mesh_file = mesh_root / object_name / "mesh.obj"
    if not mesh_file.exists():
        return None
    mesh = trimesh.load(mesh_file, process=False)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)


def extract_rot_trans_arti(processed_root: Path, output_root: Path, mesh_root: Path, overwrite: bool) -> None:
    """Extract per-sequence ARCTIC object metadata from the raw per-frame dumps.

    For every subject/sequence ``.npy`` file under `processed_root` this
    writes, into `output_root/<subject>/`, the raw fields read directly by
    ``dataset_ARCTIC.py`` (rotation, translation, articulation, vertex
    arrays, ``K_ego.npy``, ...) plus two derived quantities it also needs:

    * ``{name}_rot_local_world.npy`` / ``_trans_local_world.npy`` /
      ``_scale_local_world.npy`` -- the similarity transform (R, t, s) that
      maps the object's articulated template mesh onto the WORLD-frame
      object vertices for each frame (ports ``local_world_R_T_S`` from
      extract_rot_trans_arti.py).
    * ``{name}_bbox3d.npy`` -- the 3D bbox corners projected into the ego
      view and rescaled to crop coordinates (ports ``extract_bbox`` /
      ``transform_bbox3d_to_crop`` from the same reference script).
    """
    output_root.mkdir(parents=True, exist_ok=True)
    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for subject_dir in _subjects(processed_root):
        output_dir = output_root / subject_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        k_ego_written = (output_dir / "K_ego.npy").exists() and not overwrite
        for source_file in sorted(subject_dir.glob("*.npy")):
            name = source_file.stem
            try:
                data = np.load(source_file, allow_pickle=True).item()
                params = data["params"]
                cam = data["cam_coord"]
                world = data["world_coord"]
            except (KeyError, ValueError, AttributeError) as error:
                print(f"[metadata] skipping {source_file}: {error}")
                continue

            values = {
                f"{name}_rot.npy": params.get("obj_rot"),
                f"{name}_trans.npy": params.get("obj_trans"),
                f"{name}_arti.npy": params.get("obj_arti"),
                f"{name}_world_verts.npy": world.get("verts.object"),
                f"{name}_world2ego.npy": params.get("world2ego"),
                f"{name}_verts_left.npy": world.get("verts.left"),
                f"{name}_verts_right.npy": world.get("verts.right"),
                f"{name}_verts_left_camera.npy": cam.get("verts.left"),
                f"{name}_verts_right_camera.npy": cam.get("verts.right"),
                f"{name}_obj_cam_verts.npy": cam.get("verts.object"),
            }
            for filename, value in values.items():
                destination = output_dir / filename
                if value is not None and (overwrite or not destination.exists()):
                    np.save(destination, np.asarray(value))
            if not k_ego_written and params.get("K_ego") is not None:
                np.save(output_dir / "K_ego.npy", np.asarray(params["K_ego"]))
                k_ego_written = True

            bbox3d_2d = data.get("2d", {}).get("bbox3d")
            bbox3d_file = output_dir / f"{name}_bbox3d.npy"
            if bbox3d_2d is not None and (overwrite or not bbox3d_file.exists()):
                # Ego view only (view_idx 0): the reference transform just
                # rescales the projected corners for this view.
                bbox3d = np.asarray(bbox3d_2d)[:, 0] * EGO_BBOX_SCALE
                np.save(bbox3d_file, bbox3d.astype(np.float32))

            arti = values[f"{name}_arti.npy"]
            world_verts = values[f"{name}_world_verts.npy"]
            object_name = name.split("_")[0]
            if object_name not in mesh_cache:
                mesh_cache[object_name] = _load_mesh(mesh_root, object_name)
            mesh_entry = mesh_cache[object_name]
            result_file = output_dir / f"{name}_rot_local_world.npy"
            if arti is None or world_verts is None or mesh_entry is None or (result_file.exists() and not overwrite):
                continue
            template, faces = mesh_entry
            parts_file = mesh_root / object_name / "parts.json"
            rotations, translations, scales = [], [], []
            for angle, target in zip(np.asarray(arti).reshape(-1), np.asarray(world_verts)):
                articulated = _articulate(template, faces, parts_file, float(angle))
                rotation, translation, scale = _similarity(articulated, target)
                rotations.append(rotation)
                translations.append(translation)
                scales.append(scale)
            np.save(result_file, np.asarray(rotations))
            np.save(output_dir / f"{name}_trans_local_world.npy", np.asarray(translations))
            np.save(output_dir / f"{name}_scale_local_world.npy", np.asarray(scales))
            print(f"[metadata] {subject_dir.name}/{name}")


def extract_contacts(processed_root: Path, images_root: Path, contacts_root: Path, threshold: float, jobs: int) -> None:
    sys.path.insert(0, str(HERE / "arctic_preprocess"))
    from extract_contacts_manually import process_sequence  # existing contact implementation

    with open(Path(MANO_ROOT) / "MANO_LEFT.pkl", "rb") as handle:
        faces_left = pickle.load(handle, encoding="latin1")["f"]
    with open(Path(MANO_ROOT) / "MANO_RIGHT.pkl", "rb") as handle:
        faces_right = pickle.load(handle, encoding="latin1")["f"]
    # Only consider subjects that actually have raw per-frame dumps: some
    # subjects (e.g. s03) have resized images but were excluded upstream from
    # processed_verts/seqs, so process_sequence would just skip them anyway.
    processed_subjects = {subject.name for subject in _subjects(processed_root)}
    sequences = [
        sequence
        for subject in _subjects(images_root)
        if subject.name in processed_subjects
        for sequence in sorted(subject.iterdir())
        if sequence.is_dir()
    ]
    print(f"[contacts] processing {len(sequences)} sequences")
    if jobs == 1:
        for sequence in sequences:
            process_sequence(sequence, faces_left, faces_right, threshold, contacts_root)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=jobs)(delayed(process_sequence)(sequence, faces_left, faces_right, threshold, contacts_root) for sequence in sequences)


def resize_images(overwrite: bool) -> None:
    # The existing resize utility is GPU-capable and already enumerates s01--s10.
    if Path(IMAGES_ROOT).exists() and not overwrite:
        print(f"[resize] {IMAGES_ROOT} exists; existing object folders are skipped by resize224.py")
    subprocess.run([sys.executable, str(HERE / "arctic_preprocess" / "resize224.py")], check=True)


def render_segmentations(processed_root: Path, segment_root: Path) -> None:
    """Render per-frame hand/object segmentation masks via the ARCTIC visualizer.

    Shells out to ``ARCTIC_ROOT/scripts_data/visualizer.py`` exactly as
    ``run_arctic_segmentation.py`` did (same flags, same output-folder
    naming: ``f"{subject}_{sequence}_0"``). ``dataset_ARCTIC.py`` reads masks
    from ``segment_root/<that name>/images/mask_{left,right,object}/*.png``,
    so each render is checked against that path: the checked-out visualizer
    can drift (e.g. write to a differently-named or relative output folder)
    without raising an error, and that failure should not be silent here.
    """
    visualizer = Path(ARCTIC_ROOT) / "scripts_data" / "visualizer.py"
    if not visualizer.exists():
        raise FileNotFoundError(f"ARCTIC visualizer not found: {visualizer}")
    for subject in _subjects(processed_root):
        for sequence in sorted(subject.glob("*.npy")):
            out_name = f"{subject.name}_{sequence.stem}_0"
            print(f"[segmentation] {out_name}")
            subprocess.run(
                [sys.executable, str(visualizer), "--seq_p", str(sequence), "--object", "--mano", "--headless", "--view_idx", "0"],
                cwd=ARCTIC_ROOT,
                check=True,
            )
            expected = Path(segment_root) / out_name / "images"
            if not expected.is_dir():
                print(
                    f"[segmentation] WARNING: expected output not found at {expected}; "
                    f"check {visualizer} for local edits to its render_seq() out_folder."
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=("resize", "contacts", "metadata", "segmentation"), default=("resize", "contacts", "metadata", "segmentation"))
    parser.add_argument("--jobs", type=int, default=1, help="Parallel ARCTIC contact workers (default: 1).")
    parser.add_argument("--threshold", type=float, default=0.0025)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--processed-root", type=Path, default=Path(PROCESSED_SEQS_ROOT))
    parser.add_argument("--images-root", type=Path, default=Path(IMAGES_ROOT))
    parser.add_argument("--contacts-root", type=Path, default=Path(CONTACTS_ROOT))
    parser.add_argument("--metadata-root", type=Path, default=Path(ROT_TRANS_ARTI_ROOT))
    parser.add_argument("--mesh-root", type=Path, default=Path(MESH_ROOT))
    parser.add_argument("--segment-root", type=Path, default=Path(SEGMENT_ROOT))
    args = parser.parse_args()
    if "resize" in args.stages:
        resize_images(args.overwrite)
    if "contacts" in args.stages:
        extract_contacts(args.processed_root, args.images_root, args.contacts_root, args.threshold, args.jobs)
    if "metadata" in args.stages:
        extract_rot_trans_arti(args.processed_root, args.metadata_root, args.mesh_root, args.overwrite)
    if "segmentation" in args.stages:
        render_segmentations(args.processed_root, args.segment_root)


if __name__ == "__main__":
    main()
