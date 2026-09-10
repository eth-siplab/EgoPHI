import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torch_geometric.utils import to_undirected
from torchvision.io import read_image
from torchvision.transforms.functional import to_tensor

import config


def get_total_memory_usage_mb():
    parent = psutil.Process(os.getpid())
    mem = parent.memory_info().rss
    for child in parent.children(recursive=True):
        mem += child.memory_info().rss
    return mem / 1024**2

class ArcticSequenceDataset(Dataset):
    def __init__(self,
                 images_root=config.IMAGES_ROOT,
                 segment_root=config.SEGMENT_ROOT,
                 contacts_root=config.CONTACTS_ROOT,
                 distances_root=config.DISTANCES_ROOT,
                 force_root=config.FORCE_ROOT,
                 processed_seqs_root=config.PROCESSED_SEQS_ROOT,
                 mesh_root=config.MESH_ROOT,
                 GT_mano_root=config.GT_MANO_ROOT,
                 hamer_mano_root=config.HAMER_MANO_ROOT,
                 processed_force_root=config.PROCESSED_FORCE_ROOT,
                 cameras=(0,),
                 sequence_length=128,
                 overlap=0,
                 load_hamer=False,
                 participants=None,
                 device=None):
        """
        images_root: path to cropped_images
        contacts_root: path to processed hand-object contacts
        distances_root: path to processed distances
        cameras: list of camera indices to use (e.g. [0,1,2])
        sequence_length: how many frames per batch (default 128)
        transform: optional torchvision transform
        hamer_mano_root: path to precomputed HAMER MANO parameters
        processed_force_root: path to the precomputed per-vertex force
            magnitude/direction arrays produced by precompute_forces_arctic.py.
            force_root's raw per-frame files are still scanned in __init__ to
            build to_skip/cached_forces (used throughout this class for frame
            reindexing, not just forces), but the forces returned by
            _load_contacts are read from here instead of being normalized
            on the fly.
        load_hamer: if True, __getitem__ also loads and returns the raw
            HAMER MANO parameters (pose/rot/shape/trans for both hands) and
            the source image path. Off by default so training (which
            doesn't need these) doesn't pay for the extra file reads --
            evaluation scripts should pass load_hamer=True.
        participants: optional list of participant IDs (e.g. ['s05']) to
            restrict indexing to. __init__ scans the whole dataset on disk
            before any train/val split happens, which is fine for training
            (paid once, amortized over hours) but very slow for evaluation
            if you only need one participant -- pass participants=[...] to
            skip scanning everyone else. Defaults to None (scan all
            participants, i.e. unchanged behavior).
        """
        self.images_root = images_root
        self.segment_root = segment_root
        self.contacts_root = contacts_root
        self.distances_root = distances_root
        self.force_root = force_root
        self.processed_seqs_root = processed_seqs_root
        self.mesh_root = mesh_root
        self.GT_mano_root = GT_mano_root
        self.processed_force_root = processed_force_root
        self.cameras = [str(c) for c in cameras]
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.subj_path = config.ROT_TRANS_ARTI_ROOT
        self.hamer_mano_root = hamer_mano_root
        self.hamer_verts_root = config.HAMER_VERTS_ROOT
        self.load_hamer = load_hamer
        self._participants_filter = set(participants) if participants is not None else None
        self.min_max_magnitude_path = config.FORCE_MIN_MAX_MAGNITUDE_ROOT
        self.to_skip = []
        json_path = config.SOFA_FORCE_LOG_MAGNITUDE_STATS_JSON
        self.haco_root = config.HACO_CONTACTS_ROOT

        with open(json_path, "r") as f:
            self.stats = json.load(f)

        self.cached_contacts = {}
        self.cached_forces = {}

        # NOTE: this loop intentionally does NOT use the `participants` filter,
        # even though it's the slowest of the four __init__ scans. It builds
        # self.to_skip, and _index_sequences() later computes each object's
        # valid frame count as `len(forces) - len(self.to_skip)` -- a GLOBAL
        # count, not a per-object one. Restricting this loop to fewer
        # participants shrinks that global count and desyncs it from the
        # actual number of cached segmentation masks per object, causing an
        # IndexError in _load_ARCTIC_segmentations. Fixing the underlying
        # per-object counting bug would change which frames training sees,
        # so it's left as-is; only the scan below is exempted from filtering.
        for participant in sorted(os.listdir(self.images_root)):
            if not participant.startswith('s') or participant == 's03':
                continue
            participant_path = os.path.join(self.images_root, participant)
            if not os.path.isdir(participant_path):
                continue

            for object_name in sorted(os.listdir(participant_path)):
                object_path = os.path.join(participant_path, object_name)
                if not os.path.isdir(object_path):
                    continue

                contacts_dict = {}
                forces_dict = {}

                for suffix in ['left', 'right', 'object']:
                    contact_path = os.path.join(self.contacts_root, participant, f"{object_name}_{suffix}.npy")
                    contacts_dict[suffix] = contact_path

                    force_folder = os.path.join(self.force_root, participant, f"{object_name}_{suffix}")
                    force_files = []
                    all_files = sorted(os.listdir(force_folder))

                    for f in all_files:
                        if not f.startswith("forces_") or not f.endswith(".npy"):
                            continue
                        file_path = os.path.join(force_folder, f)
                        # NOTE: the file is appended unconditionally (even when it turns
                        # out to be NaN/Inf below). This matches the legacy loader, whose
                        # `_index_sequences` counts frames as
                        # `len(cached_forces[...]['left']) - len(self.to_skip)` -- i.e. it
                        # expects `cached_forces` to include the bad files and rely on the
                        # global `to_skip` count to size things down. Only appending the
                        # good files here (as an earlier version of this file did) desyncs
                        # that arithmetic from the legacy behavior and shifts every
                        # per-object frame count.
                        force_files.append(file_path)
                        try:
                            arr = np.load(file_path)
                            if np.isfinite(arr).all():
                                continue
                            else:
                                parts = file_path.split(os.sep)
                                skipped_participant = parts[-3]
                                skipped_object = parts[-2].replace(f"_{suffix}", "")
                                frame_id = int(os.path.basename(file_path).split("_")[1].split(".")[0])
                                self.to_skip.append((skipped_participant, skipped_object, frame_id))
                        except Exception as e:
                            parts = file_path.split(os.sep)
                            skipped_participant = parts[-3]
                            skipped_object = parts[-2].replace(f"_{suffix}", "")
                            frame_id = int(os.path.basename(file_path).split("_")[1].split(".")[0])
                            self.to_skip.append((skipped_participant, skipped_object, frame_id))

                    force_files = sorted(
                        force_files,
                        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
                    )
                    forces_dict[suffix] = force_files

                self.cached_contacts[(participant, object_name)] = contacts_dict
                self.cached_forces[(participant, object_name)] = forces_dict

        skipped_set = set(self.to_skip)

        self.segmentation_cache = {}
        for participant in self._list_participants(self.images_root):
            if not participant.startswith('s') or participant == 's03':
                continue
            for object_name in sorted(os.listdir(os.path.join(self.images_root, participant))):
                for cam in self.cameras:
                    obj_folder = f"{participant}_{object_name}_{cam}"
                    seg_root = os.path.join(self.segment_root, obj_folder, "images")

                    seg_masks = {}
                    for suffix in ['left', 'right', 'object']:
                        mask_folder = os.path.join(seg_root, f"mask_{suffix}")
                        all_masks = sorted(glob.glob(os.path.join(mask_folder, "*.png")))

                        filtered_masks = []
                        for path in all_masks:
                            frame_id = int(os.path.basename(path).split(".")[0])
                            if (participant, object_name, frame_id) not in skipped_set:
                                filtered_masks.append(path)

                        seg_masks[suffix] = filtered_masks

                    self.segmentation_cache[(participant, object_name, cam)] = seg_masks

        # haco_contacts_cache would back _load_HACO_contacts, but that method
        # is never called (see the commented-out call in __getitem__) -- so
        # skip the expensive walk over self.haco_root and just keep the
        # attribute present in case it's wired back up later.
        self.haco_contacts_cache = {}

        self.cached_distances = {}
        for participant in self._list_participants(self.images_root):
            if not participant.startswith('s') or participant == 's03':
                continue
            participant_path = os.path.join(self.images_root, participant)
            if not os.path.isdir(participant_path):
                continue

            for object_name in sorted(os.listdir(participant_path)):
                object_path = os.path.join(participant_path, object_name)
                if not os.path.isdir(object_path):
                    continue

                dists_dict = {}
                for suffix in ['dists_norm_left', 'dists_norm_right', 'dists_norm_obj_left', 'dists_norm_obj_right']:
                    npy_path = os.path.join(self.distances_root, participant, f"{object_name}_1_{suffix}.npy")
                    if os.path.isfile(npy_path):
                        dists_dict[suffix] = npy_path
                    else:
                        print(f"Warning: {npy_path} not found, skipping")
                self.cached_distances[(participant, object_name)] = dists_dict

        self.samples = self._index_sequences()
        self.mesh_cache = {}
        object_names = set(sample['object'] for sample in self.samples)
        for obj_name in object_names:
            coords, faces, edge_index, edge_weight, part_ids, pivot, vertex_normals = self._load_articulated_mesh(obj_name, hinge_weight=0.1)
            self.mesh_cache[obj_name] = {
                'coords': coords,                 # [num_verts, 3]
                'faces': torch.tensor(faces),     # [num_faces, 3]
                'edge_index': edge_index,         # [2, num_edges]
                'edge_weight': edge_weight,       # [num_edges]
                'part_ids': part_ids,             # [num_verts]
                'pivot': pivot,                   # [3]
                'vertex_normals': vertex_normals  # [num_verts, 3]
            }

        self.cached_images = {}
        for (participant, object_name, cam), paths in self.cached_image_paths.items():
            self.cached_images[(participant, object_name, cam)] = paths


    def _get_GT_valid_frame_indices(self, participant, object_name):
        """
        Returns all valid frame indices (as ints) for GT MANO params.
        Since GT MANO params are complete, all indices are considered valid.
        """
        folder = os.path.join(self.gt_mano_root, participant)
        if not os.path.isdir(folder):
            return []

        files = glob.glob(os.path.join(folder, object_name, '*.pt'))
        candidate_indices = set()

        for f in files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            frame_str = parts[0]
            if frame_str.isdigit():
                candidate_indices.add(int(frame_str))

        return sorted(candidate_indices)


    def _list_participants(self, root):
        """
        Lists participant folders under `root`, applying the constructor's
        `participants` filter if one was given.
        """
        names = sorted(os.listdir(root))
        if self._participants_filter is not None:
            names = [n for n in names if n in self._participants_filter]
        return names

    def _index_sequences(self):
        samples = []
        skipped_set = set(self.to_skip)

        self.cached_image_paths = {}  # (participant, object_name, cam) -> sorted list of paths
        for participant in self._list_participants(self.images_root):
            participant_path = os.path.join(self.images_root, participant)
            if not os.path.isdir(participant_path) or participant=='s03':
                continue

            for object_name in sorted(os.listdir(participant_path)):
                object_path = os.path.join(participant_path, object_name)
                if not os.path.isdir(object_path):
                    continue

                frame_counts = []
                for cam in self.cameras:
                    cam_folder = os.path.join(object_path, cam)
                    if not os.path.isdir(cam_folder):
                        break
                    all_jpgs = sorted(glob.glob(os.path.join(cam_folder, '*.jpg')))

                    filtered_jpgs = []
                    for path in all_jpgs:
                        frame_id = int(os.path.basename(path).split(".")[0])
                        if (participant, object_name, frame_id) not in skipped_set:
                            filtered_jpgs.append(path)

                    self.cached_image_paths[(participant, object_name, cam)] = filtered_jpgs

                    # Legacy quirk (kept for exact parity): this subtracts the GLOBAL
                    # `to_skip` count (bad force frames across ALL participants/objects)
                    # from THIS object's total force-file count, not a per-object count.
                    # It under-counts the usable frames for every object by the same
                    # global amount, truncating the tail of each sequence. It doesn't
                    # affect *which* frame maps to which index (that's handled by the
                    # per-object `self.to_skip` filtering in `_load_obj_R_T_arti` /
                    # `_load_contacts`), only how many total (start_idx, length) chunks
                    # get created per object.
                    valid_indices = list(range(
                        len(self.cached_forces[(participant, object_name)]['left']) - len(self.to_skip)
                    ))
                    if len(valid_indices) == 0:
                        continue
                    print('VALID VERTICES    ', len(valid_indices))

                    min_frames = len(valid_indices)
                    frame_counts.append(min_frames)

                    contact_ok, contact_path = self._check_contact_files(participant, object_name)
                    distance_ok, distance_path = self._check_distance_files(participant, object_name)
                    proc_seqs_ok, proc_seqs_path = self._check_processed_seqs(participant, object_name)

                    if not contact_ok or not distance_ok or not proc_seqs_ok:
                        break

                    if min_frames < self.sequence_length:
                        # Not enough frames for a full chunk: keep it short and pad in __getitem__.
                        samples.append({
                            'participant': participant,
                            'object': object_name,
                            'start_idx': 0,
                            'length': min_frames,
                            'frame_indices': valid_indices,
                            'total_frame_num': min_frames,
                            'proc_seqs_path': proc_seqs_path
                        })

                    else:
                        if self.overlap == 0:
                            num_chunks = min_frames // self.sequence_length
                            for chunk_idx in range(num_chunks):
                                start_idx = chunk_idx * self.sequence_length
                                samples.append({
                                    'participant': participant,
                                    'object': object_name,
                                    'start_idx': start_idx,
                                    'length': self.sequence_length,
                                    'frame_indices': valid_indices,
                                    'proc_seqs_path': proc_seqs_path,
                                    'total_frame_num': min_frames
                                })
                        else:
                            step = self.sequence_length - self.overlap
                            for start_idx in range(0, min_frames - self.sequence_length + 1, step):
                                samples.append({
                                    'participant': participant,
                                    'object': object_name,
                                    'start_idx': start_idx,
                                    'length': self.sequence_length,
                                    'frame_indices': valid_indices,
                                    'proc_seqs_path': proc_seqs_path,
                                    'total_frame_num': min_frames
                                })

        return samples

    def _check_contact_files(self, participant, object_name):
        required_suffixes = [
            'left', 'right', 'object'
        ]
        for suffix in required_suffixes:
            path = os.path.join(self.contacts_root, participant, f"{object_name}_{suffix}.npy")
            if not os.path.isfile(path):
                return False, path
        return True, path

    def _check_distance_files(self, participant, object_name):
        required_suffixes = [
            'dists_norm_left', 'dists_norm_right',
            'dists_norm_obj_left', 'dists_norm_obj_right'
        ]
        for suffix in required_suffixes:
            path = os.path.join(self.distances_root, participant, f"{object_name}_1_{suffix}.npy")
            if not os.path.isfile(path):
                return False, path
        return True, path

    def _check_processed_seqs(self, participant, object_name):
        path = os.path.join(self.processed_seqs_root, participant, f"{object_name}.npy")
        if not os.path.isfile(path):
            return False, path
        return True, path

    def __len__(self):
        return len(self.samples)

    def _load_images(self, participant, object_name, frame_indices):
        frames = []

        for i in frame_indices:
            frame_views = []
            for cam in self.cameras:
                img_path = self.cached_images[(participant, object_name, cam)][i]
                img = read_image(img_path).float() / 255.0
                frame_views.append(img)

            frame_views = torch.stack(frame_views)  # [num_cams, C, H, W]
            frames.append(frame_views)

        return torch.stack(frames), img_path  # [seq_len, num_cams, C, H, W], last frame's image path


    def _load_cam_int(self, participant):
        K_ego_path = os.path.join(self.subj_path, participant, "K_ego.npy")
        K_ego = np.load(K_ego_path)[0]
        return torch.tensor(K_ego)

    def _load_obj_R_T_arti(self, participant, object, start_idx, length):
        rot_path = os.path.join(self.subj_path, participant, f"{object}_rot_local_world.npy")
        trans_path = os.path.join(self.subj_path, participant, f"{object}_trans_local_world.npy")
        scale_path = os.path.join(self.subj_path, participant, f"{object}_scale_local_world.npy")
        arti_path = os.path.join(self.subj_path, participant, f"{object}_arti.npy")
        world_verts_path = os.path.join(self.subj_path, participant, f"{object}_obj_cam_verts.npy")
        left_verts_path = os.path.join(self.subj_path, participant, f"{object}_verts_left_camera.npy")
        right_verts_path = os.path.join(self.subj_path, participant, f"{object}_verts_right_camera.npy")
        world2ego_path = os.path.join(self.subj_path, participant, f"{object}_world2ego.npy")
        bbox3d_path = os.path.join(self.subj_path, participant, f"{object}_bbox3d.npy")

        R_full = np.load(rot_path)
        T_full = np.load(trans_path)
        s_full = np.load(scale_path)
        arti_full = np.load(arti_path)
        world_verts_full = np.load(world_verts_path)[:,0]
        left_verts_full = np.load(left_verts_path)[:,0]
        right_verts_full = np.load(right_verts_path)[:,0]
        world2ego_full = np.load(world2ego_path)
        bbox3d_full = np.load(bbox3d_path)

        hamer_dir = os.path.join(self.hamer_verts_root, object, "0")
        hamer_left_path = os.path.join(hamer_dir, f"vert_0.0_{start_idx:05d}.npy")
        hamer_right_path = os.path.join(hamer_dir, f"vert_1.0_{start_idx:05d}.npy")

        num_frames = R_full.shape[0]
        valid_frame_ids = [i for i in range(num_frames) if (participant, object, i) not in self.to_skip]
        valid_frame_ids = np.array(valid_frame_ids, dtype=int)

        R_clean = R_full[valid_frame_ids]
        T_clean = T_full[valid_frame_ids]
        s_clean = s_full[valid_frame_ids]
        arti_clean = arti_full[valid_frame_ids]
        world_verts_clean = world_verts_full[valid_frame_ids]
        left_verts_clean = left_verts_full[valid_frame_ids]
        right_verts_clean = right_verts_full[valid_frame_ids]
        world2ego_clean = world2ego_full[valid_frame_ids]
        bbox3d_clean = bbox3d_full[valid_frame_ids]

        R = R_clean[start_idx : start_idx + length]
        T = T_clean[start_idx : start_idx + length]
        s = s_clean[start_idx : start_idx + length]
        arti = arti_clean[start_idx : start_idx + length]
        world_verts = world_verts_clean[start_idx : start_idx + length]
        left_verts = left_verts_clean[start_idx : start_idx + length]
        right_verts = right_verts_clean[start_idx : start_idx + length]
        world2ego = world2ego_clean[start_idx : start_idx + length]
        bbox3d = bbox3d_clean[start_idx : start_idx + length]

        if os.path.isfile(hamer_left_path):
            hamer_left = np.load(hamer_left_path)
            hamer_left = hamer_left + left_verts.mean(axis=0) - hamer_left.mean(axis=0)
        else:
            hamer_left = np.zeros_like(left_verts)
        if os.path.isfile(hamer_right_path):
            hamer_right = np.load(hamer_right_path)
            hamer_right = hamer_right + right_verts.mean(axis=0) - hamer_right.mean(axis=0)
        else:
            hamer_right = np.zeros_like(right_verts)

        return (
            torch.tensor(R, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(arti, dtype=torch.float32),
            torch.tensor(world_verts, dtype=torch.float32),
            torch.tensor(left_verts, dtype=torch.float32),
            torch.tensor(right_verts, dtype=torch.float32),
            torch.tensor(world2ego, dtype=torch.float32),
            torch.tensor(bbox3d, dtype=torch.float32),
            torch.tensor(hamer_left, dtype=torch.float32),
            torch.tensor(hamer_right, dtype=torch.float32),
        )

    def _load_segmentations(self, participant, object_name, frame_indices):
        twohands_out = []
        cb_out = []
        obj1_out = []

        twohands = {}
        cb = {}
        obj1 = {}
        for cam in self.cameras:
            pred_twohands = os.path.join(self.segment_root, participant, object_name, cam, 'pred_twohands')
            pred_cb = os.path.join(self.segment_root, participant, object_name, cam, 'pred_cb')
            pred_obj1 = os.path.join(self.segment_root, participant, object_name, cam, 'pred_obj1')
            pred_twohands_images = sorted(glob.glob(os.path.join(pred_twohands, '*.png')))
            pred_cb_images = sorted(glob.glob(os.path.join(pred_cb, '*.png')))
            pred_obj1_images = sorted(glob.glob(os.path.join(pred_obj1, '*.png')))
            twohands[cam] = pred_twohands_images
            cb[cam] = pred_cb_images
            obj1[cam] = pred_obj1_images

        for i in frame_indices:
            twohands_views = []
            for cam in self.cameras:
                twohands_path = twohands[cam][i]
                with Image.open(twohands_path) as img:
                    img = torch.tensor(np.array(img))
                twohands_views.append(img)
            twohands_views = torch.stack(twohands_views)
            twohands_out.append(twohands_views)

            cb_views = []
            for cam in self.cameras:
                cb_path = cb[cam][i]
                with Image.open(cb_path) as img:
                    img = torch.tensor(np.array(img))
                cb_views.append(img)
            cb_views = torch.stack(cb_views)
            cb_out.append(cb_views)

            obj1_views = []
            for cam in self.cameras:
                obj1_path = obj1[cam][i]
                with Image.open(obj1_path) as img:
                    img = torch.tensor(np.array(img))
                obj1_views.append(img)
            obj1_views = torch.stack(obj1_views)
            obj1_out.append(obj1_views)

        return torch.stack(twohands_out), torch.stack(cb_out), torch.stack(obj1_out)


    def _load_ARCTIC_segmentations(self, participant, object_name, frame_indices):
        seg_cache = self.segmentation_cache
        left_out, right_out, obj_out = [], [], []

        for i in frame_indices:
            left_views, right_views, obj_views = [], [], []
            for cam in self.cameras:
                paths = seg_cache[(participant, object_name, cam)]

                left = to_tensor(Image.open(paths["left"][i]))
                right = to_tensor(Image.open(paths["right"][i]))
                obj = to_tensor(Image.open(paths["object"][i]))

                left_views.append(left)
                right_views.append(right)
                obj_views.append(obj)

            left_out.append(torch.stack(left_views))
            right_out.append(torch.stack(right_views))
            obj_out.append(torch.stack(obj_views))

        return torch.stack(left_out), torch.stack(right_out), torch.stack(obj_out)


    def _load_HACO_contacts(self, participant, object_name, frame_indices):
        cache = self.haco_contacts_cache
        left_out, right_out = [], []

        contact_files = cache.get((participant, object_name), {'left': [], 'right': []})
        left_files = {int(os.path.basename(f).split('_')[0]): f for f in contact_files['left']}
        right_files = {int(os.path.basename(f).split('_')[0]): f for f in contact_files['right']}

        valid_frame_indices = [idx for idx in frame_indices if (participant, object_name, idx) not in self.to_skip]

        for idx in valid_frame_indices:
            if idx in left_files:
                left_tensor = torch.load(left_files[idx])
                left_tensor = left_tensor.float()
            else:
                left_tensor = torch.zeros(778, dtype=torch.float32)

            if idx in right_files:
                right_tensor = torch.load(right_files[idx])
                right_tensor = right_tensor.float()
            else:
                right_tensor = torch.zeros(778, dtype=torch.float32)

            left_out.append(left_tensor)
            right_out.append(right_tensor)

        return torch.stack(left_out), torch.stack(right_out)

    def _load_contacts(self, participant, object_name, start_idx, length):
        """
        Contacts are read and reindexed (to_skip-filtered) here as before.
        Forces are no longer normalized on the fly -- magnitude/direction are
        read straight from precompute_forces_arctic.py's output, already
        contact-masked, normalized, and reindexed the same way.
        """
        contact_data = {}
        force_data = {}
        force_vector = {}

        contacts_dict = self.cached_contacts.get((participant, object_name), {})

        for suffix in ['left', 'right', 'object']:
            contact_path = contacts_dict.get(suffix, None)
            if contact_path is None or not os.path.isfile(contact_path):
                print(f"Warning: contact path {contact_path} not found, skipping")
                continue

            arr_contact = np.load(contact_path, mmap_mode='r')
            valid_frame_ids = [i for i in range(len(arr_contact)) if (participant, object_name, i) not in self.to_skip]
            valid_frame_ids = np.array(valid_frame_ids, dtype=int)
            arr_contact_clean = arr_contact[valid_frame_ids]

            contact_data[suffix] = torch.from_numpy(arr_contact_clean[start_idx:start_idx+length]).float()

            mag_path = os.path.join(self.processed_force_root, participant, f"{object_name}_{suffix}_magnitude.npy")
            dir_path = os.path.join(self.processed_force_root, participant, f"{object_name}_{suffix}_direction.npy")
            if not (os.path.isfile(mag_path) and os.path.isfile(dir_path)):
                continue

            magnitude = np.load(mag_path, mmap_mode='r')
            direction = np.load(dir_path, mmap_mode='r')
            force_data[suffix] = torch.from_numpy(np.asarray(magnitude[start_idx:start_idx+length])).float()
            force_vector[suffix] = torch.from_numpy(np.asarray(direction[start_idx:start_idx+length])).float()

        return contact_data, force_data, force_vector


    def _load_distances(self, participant, object_name, start_idx, length):
        distance_data = {}
        dists_dict = self.cached_distances.get((participant, object_name), {})

        for suffix, npy_path in dists_dict.items():
            if not os.path.isfile(npy_path):
                print(f"Warning: {npy_path} not found, skipping")
                continue
            arr = np.load(npy_path, mmap_mode='r')
            distance_data[suffix] = torch.from_numpy(arr[start_idx:start_idx+length]).float()

        return distance_data


    def stack_all(self, *args):
        return [torch.stack(x, dim=1) for x in args]


    def _load_GT_proc_seqs(self, participant, object_name, start_idx, length):
        folder = os.path.join(self.GT_mano_root, participant)

        pose_r = torch.load(os.path.join(folder, f"{object_name}_pose_r.pt"))
        num_f = pose_r.shape[0]

        valid_frame_ids = [i for i in range(num_f) if (participant, object_name, i) not in self.to_skip]
        valid_frame_ids = torch.tensor(valid_frame_ids, dtype=torch.long)

        pose_r = torch.load(os.path.join(folder, f"{object_name}_pose_r.pt"))[valid_frame_ids]
        rot_r = torch.load(os.path.join(folder, f"{object_name}_rot_r_cam.pt"))[valid_frame_ids]
        shape_r = torch.load(os.path.join(folder, f"{object_name}_shape_r.pt"))[valid_frame_ids]
        trans_r = torch.load(os.path.join(folder, f"{object_name}_trans_r.pt"))[valid_frame_ids]
        theta_r = torch.cat([rot_r, pose_r], dim=1)

        pose_l = torch.load(os.path.join(folder, f"{object_name}_pose_l.pt"))[valid_frame_ids]
        rot_l = torch.load(os.path.join(folder, f"{object_name}_rot_l_cam.pt"))[valid_frame_ids]
        shape_l = torch.load(os.path.join(folder, f"{object_name}_shape_l.pt"))[valid_frame_ids]
        trans_l = torch.load(os.path.join(folder, f"{object_name}_trans_l.pt"))[valid_frame_ids]
        theta_l = torch.cat([rot_l, pose_l], dim=1)

        theta_r = theta_r[start_idx:start_idx+length]
        theta_l = theta_l[start_idx:start_idx+length]
        shape_r = shape_r[start_idx:start_idx+length]
        shape_l = shape_l[start_idx:start_idx+length]
        trans_r = trans_r[start_idx:start_idx+length]
        trans_l = trans_l[start_idx:start_idx+length]

        return theta_r, theta_l, shape_r, shape_l, trans_r, trans_l

    def _load_HAMER_verts_flexible(self, participant, object_name, frame_indices):
        """
        Loads raw (per-frame) HAMER MANO parameters for both hands. Falls
        back to zeros for a hand/frame with no HAMER output on disk, so a
        missing detection never crashes the loader -- only used when
        load_hamer=True (evaluation).
        """
        pose_l, pose_r, rot_l, rot_r, shape_r, shape_l, trans_r, trans_l = [], [], [], [], [], [], [], []

        ZERO_POSE = torch.zeros(135)
        ZERO_ROT = torch.zeros(3, 3)
        ZERO_SHAPE = torch.zeros(10)
        ZERO_TRANS = torch.zeros(3)

        for cam in self.cameras:
            folder = os.path.join(self.hamer_mano_root, participant, object_name, cam)
            values = {key: [] for key in ('pose_r', 'rot_r', 'shape_r', 'trans_r',
                                          'pose_l', 'rot_l', 'shape_l', 'trans_l')}
            for frame_index in frame_indices:
                index_str = f"{frame_index:05d}"
                for side, suffixes in (
                    ('r', ['1_hand_pose.npy', '1_global_orient.npy', '1_betas.npy', '1_transl.npy']),
                    ('l', ['0_hand_pose.npy', '0_global_orient.npy', '0_betas.npy', '0_transl.npy']),
                ):
                    paths = [os.path.join(folder, f"{index_str}_{suffix}") for suffix in suffixes]
                    if all(os.path.exists(path) for path in paths):
                        loaded = [torch.tensor(np.load(path), dtype=torch.float32) for path in paths]
                    else:
                        loaded = [ZERO_POSE.clone(), ZERO_ROT.clone(), ZERO_SHAPE.clone(), ZERO_TRANS.clone()]
                    for key, value in zip((f'pose_{side}', f'rot_{side}', f'shape_{side}', f'trans_{side}'), loaded):
                        values[key].append(value)

            pose_r.append(torch.stack(values['pose_r']))
            rot_r.append(torch.stack(values['rot_r']))
            shape_r.append(torch.stack(values['shape_r']))
            trans_r.append(torch.stack(values['trans_r']))
            pose_l.append(torch.stack(values['pose_l']))
            rot_l.append(torch.stack(values['rot_l']))
            shape_l.append(torch.stack(values['shape_l']))
            trans_l.append(torch.stack(values['trans_l']))

        return self.stack_all(pose_r, rot_r, pose_l, rot_l, shape_r, shape_l, trans_r, trans_l)

    def _align_HAMER_verts(self,
                        pose_r, rot_r, pose_l, rot_l,
                        shape_r, shape_l, trans_r, trans_l,
                        gt_left_verts_all, gt_right_verts_all, mano_layer_left, mano_layer_right):
        """
        Runs MANO on the raw HAMER parameters and rigidly translates the
        resulting hand meshes so their centers match the GT hand meshes
        (HAMER's absolute translation is not metrically accurate; only the
        pose/shape are used, the position is corrected against GT).
        """
        import pytorch3d.transforms as tf
        th_pose_l = pose_l.float().reshape(-1, 15, 3, 3)
        th_rot_l = rot_l.float().reshape(-1, 3, 3)
        th_shape_l = shape_l.float().reshape(-1, 10)
        th_trans_l = trans_l.float().reshape(-1, 3)

        th_pose_r = pose_r.float().reshape(-1, 15, 3, 3)
        th_rot_r = rot_r.float().reshape(-1, 3, 3)
        th_shape_r = shape_r.float().reshape(-1, 10)
        th_trans_r = trans_r.float().reshape(-1, 3)

        th_gt_left_verts = gt_left_verts_all.float().reshape(-1, gt_left_verts_all.shape[-2], 3)
        th_gt_right_verts = gt_right_verts_all.float().reshape(-1, gt_right_verts_all.shape[-2], 3)

        th_left_hand_pose = tf.matrix_to_axis_angle(th_pose_l)
        th_left_gl_ori = tf.matrix_to_axis_angle(th_rot_l)

        th_left_gl_ori[:, 1:] = th_left_gl_ori[:, 1:] * -1
        th_left_hand_pose[:, :, 1:] = th_left_hand_pose[:, :, 1:] * -1
        th_left_pose_coeffs = torch.cat([th_left_gl_ori.unsqueeze(1), th_left_hand_pose], dim=1).reshape(-1, 48).to(pose_r.device)

        th_verts_left_mm, _ = mano_layer_left(
            th_pose_coeffs=th_left_pose_coeffs,
            th_betas=th_shape_l,
            th_trans=th_trans_l,
        )

        th_right_pose_coeffs = torch.cat([th_rot_r.unsqueeze(1), th_pose_r], dim=1).to(pose_r.device)
        th_verts_right_mm, _ = mano_layer_right(
            th_pose_coeffs=th_right_pose_coeffs, th_betas=th_shape_r.to(pose_r.device), th_trans=th_trans_r.to(pose_r.device)
        )

        # HAMER outputs millimeters; convert to meters before rigidly
        # translating to align with the GT hand centers.
        th_verts_left_m = th_verts_left_mm / 1000.0
        th_verts_right_m = th_verts_right_mm / 1000.0

        hamer_l_min = torch.min(th_verts_left_m, dim=1, keepdim=True).values
        hamer_l_max = torch.max(th_verts_left_m, dim=1, keepdim=True).values
        hamer_left_center_m = (hamer_l_min + hamer_l_max) / 2.0

        hamer_r_min = torch.min(th_verts_right_m, dim=1, keepdim=True).values
        hamer_r_max = torch.max(th_verts_right_m, dim=1, keepdim=True).values
        hamer_right_center_m = (hamer_r_min + hamer_r_max) / 2.0

        gt_left_center_m = th_gt_left_verts.mean(dim=1).unsqueeze(1)
        gt_right_center_m = th_gt_right_verts.mean(dim=1).unsqueeze(1)

        T_left_correction = gt_left_center_m - hamer_left_center_m
        T_right_correction = gt_right_center_m - hamer_right_center_m

        th_verts_left_aligned = th_verts_left_m + T_left_correction
        th_verts_right_aligned = th_verts_right_m + T_right_correction

        output_shape = tuple(gt_left_verts_all.shape[:-2]) + (th_verts_left_aligned.shape[-2], 3)
        return th_verts_left_aligned.reshape(output_shape), th_verts_right_aligned.reshape(output_shape)


    def _load_articulated_mesh(self, object_name, hinge_weight=0.1):
        """
        Loads a whole mesh and part labels from parts.json, and returns:
        - vertex features with part membership (coords + one-hot part ID)
        - hinge-aware edge_index and edge_weight
        """
        obj_name = object_name.split('_')[0]

        mesh_whole = trimesh.load(os.path.join(self.mesh_root, obj_name, "mesh.obj"))
        vertex_normals = torch.tensor(mesh_whole.vertex_normals)
        verts_whole = np.array(mesh_whole.vertices)
        faces_whole = np.array(mesh_whole.faces)
        pivot = torch.load(os.path.join(self.mesh_root, obj_name, "pivot.pt"))

        parts_path = os.path.join(self.mesh_root, obj_name, "parts.json")
        with open(parts_path, "r") as f:
            part_ids = torch.tensor(np.array(json.load(f), dtype=np.int64))

        assert len(part_ids) == len(verts_whole), \
            f"parts.json length {len(part_ids)} does not match mesh vertices {len(verts_whole)}"

        coords = torch.from_numpy(verts_whole).float()
        center = coords.mean(dim=0, keepdim=True)
        scale = coords.abs().max()
        coords = (coords - center)/ scale
        pivot = (pivot - center)/ scale

        # --- Build hinge-aware edges ---
        edges = np.array(mesh_whole.edges_unique)  # (E, 2)

        edge_list = []
        edge_weights = []
        for u, v in edges:
            if part_ids[u] == part_ids[v]:
                edge_list.append([u, v])
                edge_weights.append(1.0)          # rigid connection
            else:
                edge_list.append([u, v])
                edge_weights.append(hinge_weight) # hinge connection

        # --- Convert to PyTorch Geometric format ---
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)

        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)  # match undirected duplication

        return coords, faces_whole, edge_index, edge_weight, part_ids, pivot, vertex_normals

    def pad_tensor_to_length(self, in_tensor, pad_amount):
        if pad_amount <= 0:
            return in_tensor

        last = in_tensor[-1:]
        repeat_shape = [pad_amount] + [1] * (last.dim() - 1)
        pad = last.repeat(*repeat_shape)
        return torch.cat([in_tensor, pad], dim=0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        participant = sample['participant']
        object_name = sample['object']
        start_idx = sample['start_idx']
        length = sample['length']
        frame_indices = sample['frame_indices'][start_idx:start_idx+length]

        images, img_path = self._load_images(participant, object_name, frame_indices)
        left, right, obj = self._load_ARCTIC_segmentations(participant, object_name, frame_indices)
        K_ego = self._load_cam_int(participant)
        (R, T, s, arti, world_verts, left_verts, right_verts, world2ego, bbox3d,
         hamer_left, hamer_right) = self._load_obj_R_T_arti(participant, object_name, start_idx, length)
        contacts, forces, force_vector = self._load_contacts(participant, object_name, start_idx, length)
        if self.load_hamer:
            pose_r, rot_r, pose_l, rot_l, shape_r, shape_l, trans_r, trans_l = self._load_HAMER_verts_flexible(
                participant, object_name, frame_indices)
        mesh = self.mesh_cache[object_name]
        sequence_len = self.sequence_length
        template_vert = mesh['coords'].unsqueeze(0).expand(sequence_len, -1, -1)
        template_faces = mesh['faces'].unsqueeze(0).expand(sequence_len, -1, -1)
        template_edges = mesh['edge_index'].unsqueeze(0).expand(sequence_len, -1, -1)
        edge_weight = mesh['edge_weight'].unsqueeze(0).expand(sequence_len, -1)
        part_ids = mesh['part_ids'].unsqueeze(0).expand(sequence_len, -1)
        pivot = mesh['pivot'].expand(sequence_len, -1)
        vertex_normals = mesh['vertex_normals'].unsqueeze(0).expand(sequence_len, -1, -1)
        K_ego = K_ego.unsqueeze(0).expand(self.sequence_length, -1, -1)

        if length < self.sequence_length:
            pad_amount = self.sequence_length - length

            images = self.pad_tensor_to_length(images, pad_amount)
            left = self.pad_tensor_to_length(left, pad_amount)
            right = self.pad_tensor_to_length(right, pad_amount)
            obj = self.pad_tensor_to_length(obj, pad_amount)

            for key in contacts:
                last_contact = contacts[key][-1:]
                pad = last_contact.repeat(pad_amount, *[1]*(last_contact.dim()-1))
                contacts[key] = torch.cat([contacts[key], pad], dim=0)

            for key in forces:
                last_contact = forces[key][-1:]
                pad = last_contact.repeat(pad_amount, *[1]*(last_contact.dim()-1))
                forces[key] = torch.cat([forces[key], pad], dim=0)

            for key in force_vector:
                last_contact = force_vector[key][-1:]
                pad = last_contact.repeat(pad_amount, *[1]*(last_contact.dim()-1))
                force_vector[key] = torch.cat([force_vector[key], pad], dim=0)

            last_distance = R[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            R = torch.cat([R, pad_distance], dim=0)

            last_distance = T[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            T = torch.cat([T, pad_distance], dim=0)

            last_distance = s[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            s = torch.cat([s, pad_distance], dim=0)

            last_distance = arti[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            arti = torch.cat([arti, pad_distance], dim=0)

            last_distance = world_verts[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            world_verts = torch.cat([world_verts, pad_distance], dim=0)

            last_distance = left_verts[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            left_verts = torch.cat([left_verts, pad_distance], dim=0)

            last_distance = right_verts[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            right_verts = torch.cat([right_verts, pad_distance], dim=0)

            last_distance = world2ego[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            world2ego = torch.cat([world2ego, pad_distance], dim=0)

            last_distance = bbox3d[-1:]
            pad_distance = last_distance.expand(pad_amount, *last_distance.shape[1:])
            bbox3d = torch.cat([bbox3d, pad_distance], dim=0)

        item = {
            'images': images,
            'left_segmentation': left,
            'right_segmentation': right,
            'obj_segmentation': obj,
            'contacts': contacts,
            'forces' : forces,
            'force_vector' : force_vector,
            'obj_vert_template': template_vert,
            'obj_faces_template': template_faces,
            'obj_edges_template': template_edges,
            'obj_edges_weight': edge_weight,
            'obj_part_ids': part_ids,
            'pivot' : pivot,
            'vertex_normals': vertex_normals,
            'participant': participant,
            'object': object_name,
            'proc_seqs_path': sample['proc_seqs_path'],
            'K_ego': K_ego,
            'bbox3d' : bbox3d,
            'R': R,
            'T': T,
            's' : s,
            'arti': arti,
            'world_verts': world_verts,
            'left_verts': left_verts,
            'right_verts': right_verts,
            'hamer_left_aligned': hamer_left,
            'hamer_right_aligned': hamer_right,
            'world2ego' : world2ego,
            'img_path': img_path,
        }

        if self.load_hamer:
            item.update({
                'pose_r': pose_r,
                'rot_r': rot_r,
                'pose_l': pose_l,
                'rot_l': rot_l,
                'shape_r': shape_r,
                'shape_l': shape_l,
                'trans_r': trans_r,
                'trans_l': trans_l,
            })

        return item


class SameObjectBatchSampler(Sampler):
    """
    Samples batches grouped by object, compatible with DDP.
    Each process gets a non-overlapping subset of batches.
    """
    def __init__(self, object_batches, batch_size, shuffle=True, num_replicas=1, rank=0):
        self.object_batches = object_batches
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.all_batches = self._generate_batches()

    def _generate_batches(self):
        all_batches = []
        batches = self.object_batches.copy()
        if self.shuffle:
            random.shuffle(batches)

        for obj_indices in batches:
            if self.shuffle:
                random.shuffle(obj_indices)
            for i in range(0, len(obj_indices), self.batch_size):
                all_batches.append(obj_indices[i:i + self.batch_size])

        if self.shuffle:
            random.shuffle(all_batches)
        return all_batches

    def __iter__(self):
        if self.shuffle:
            self.all_batches = self._generate_batches()
        # Split batches among processes. Truncate to a multiple of
        # num_replicas (drop_last-style) so every rank gets exactly the
        # same number of batches -- otherwise a rank with fewer batches
        # finishes its loop early, stops participating in DDP's
        # backward()-triggered all-reduces, and the remaining ranks hang
        # until the NCCL watchdog times out.
        total_batches = len(self.all_batches)
        per_replica = total_batches // self.num_replicas
        usable_batches = self.all_batches[:per_replica * self.num_replicas]
        start = self.rank * per_replica
        end = start + per_replica
        return iter(usable_batches[start:end])

    def __len__(self):
        total_batches = len(self.all_batches)
        per_replica = total_batches // self.num_replicas
        return per_replica
