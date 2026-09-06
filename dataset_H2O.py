"""
H2O sequence dataset, used for evaluating the ARCTIC-trained model on the
H2O dataset. Unlike dataset_ARCTIC.py, this dataset does not touch ARCTIC
data at all -- it only indexes and loads H2O sequences.

Ground-truth hand meshes aren't available for H2O, so hand vertices come
from HAMER (an off-the-shelf hand-pose estimator) instead of GT MANO fits:
_load_obj_R_T_arti returns both the raw HAMER output (`hamer_*_orig`) and a
version rigidly translated to match the dataset's own hand-position
estimate (`hamer_*_aligned`) -- see that method for details.
"""

import glob
import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from torchvision.io import read_image

import config


def mask_forces_with_contacts(forces: np.ndarray, contacts: np.ndarray, aLL_F, all_C, total_len) -> np.ndarray:
    if forces.shape[:2] != contacts.shape[:2]:
        raise ValueError(
            f"Forces and contacts must have matching num_frames and num_vertices. "
            f"Got forces.shape={forces.shape}, contacts.shape={contacts.shape}"
            f"{aLL_F[0], aLL_F[-1]}"
            f"{all_C}"
            f"{total_len}"
        )
    # Broadcast contacts (num_frames, num_vertices, 1) -> (num_frames, num_vertices, 3)
    return forces * contacts


class H2OSequenceDataset(Dataset):
    def __init__(self,
                 images_root=config.H2O_IMAGES_ROOT,
                 contacts_root=config.H2O_CONTACTS_ROOT,
                 force_root=config.H2O_FORCE_ROOT,
                 cameras=(0,),
                 sequence_length=1,
                 overlap=0,
                 subjects=None):
        """
        images_root: H2O data root (contains subject*/h*/trial/cam*/rgb224, object_mask224, ...)
        contacts_root: root of processed H2O hand-object contacts
        force_root: root of processed H2O per-vertex forces
        cameras: list of camera indices (H2O eval currently always uses one)
        sequence_length: frames per sample (H2O eval uses 1 -- single frames)
        subjects: optional list of subject IDs (e.g. ['subject4_ego']) to
            restrict indexing to, so __init__ doesn't scan every subject on
            disk when you only need one. Defaults to None (scan everyone).
        """
        self.images_root = images_root
        self.contacts_root = contacts_root
        self.force_root = force_root
        self.rot_trans_scale_root = config.H2O_ROT_TRANS_SCALE_ROOT
        self.hamer_root = config.H2O_HAMER_ROOT

        self.cameras = [str(c) for c in cameras]
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.to_skip = []
        self._subjects_filter = set(subjects) if subjects is not None else None

        self.cached_contacts = {}
        self.cached_forces = {}

        # Traverse subjects: contacts_root/subjectN_ego/<hand_obj>/<trial>/<cam>/{suffix}_contacts.npy
        for subject in self._list_subjects(self.contacts_root):
            if not subject.startswith("subject"):
                continue
            subject_path = os.path.join(self.contacts_root, subject)
            if not os.path.isdir(subject_path):
                continue

            for root, dirs, files in os.walk(subject_path):
                contact_files = [f for f in files if f.endswith("_contacts.npy")]
                if not contact_files:
                    continue

                rel_path = os.path.relpath(root, self.contacts_root)
                if rel_path == 'subject2_ego/h2/3/cam4':
                    # known bad sequence -- excluded from processing upstream
                    continue
                sequence_id = rel_path

                contacts_dict = {}
                forces_dict = {}

                for suffix in ['left', 'right', 'object']:
                    contact_path = os.path.join(root, f"{suffix}_contacts.npy")
                    contacts_dict[suffix] = contact_path if os.path.exists(contact_path) else None

                    force_dir = os.path.join(self.force_root, rel_path, suffix)
                    if not os.path.isdir(force_dir):
                        forces_dict[suffix] = []
                        continue

                    force_files = sorted(
                        [os.path.join(force_dir, f) for f in os.listdir(force_dir)
                         if f.startswith("forces_") and f.endswith(".npy")],
                        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
                    )

                    valid_force_files = []
                    for file_path in force_files:
                        try:
                            arr = np.load(file_path)
                            if np.isfinite(arr).all():
                                valid_force_files.append(file_path)
                            else:
                                frame_id = int(os.path.basename(file_path).split("_")[1].split(".")[0])
                                self.to_skip.append((subject, sequence_id, frame_id))
                        except Exception:
                            frame_id = int(os.path.basename(file_path).split("_")[1].split(".")[0])
                            self.to_skip.append((subject, sequence_id, frame_id))
                            continue

                    forces_dict[suffix] = valid_force_files

                self.cached_contacts[(subject, sequence_id)] = contacts_dict
                self.cached_forces[(subject, sequence_id)] = forces_dict

        self.samples = self._index_sequences()

        self.cached_images = dict(self.cached_image_paths)
        self.cached_masks = dict(self.cached_mask_paths)

    def _list_subjects(self, root):
        """Lists subject folders under `root`, applying the constructor's `subjects` filter if given."""
        names = sorted(os.listdir(root))
        if self._subjects_filter is not None:
            names = [n for n in names if n in self._subjects_filter]
        return names

    def _index_sequences(self):
        samples = []
        skipped_set = set(self.to_skip)

        self.cached_image_paths = {}
        self.cached_mask_paths = {}

        # Expected structure:
        # /.../subject1_ego/h1/0/cam4/rgb224/frame_0001.jpg
        for subject in self._list_subjects(self.images_root):
            if not subject.startswith("subject"):
                continue

            subject_path = os.path.join(self.images_root, subject)
            if not os.path.isdir(subject_path):
                continue

            for hand_obj_folder in sorted(os.listdir(subject_path)):
                ho_path = os.path.join(subject_path, hand_obj_folder)
                if not os.path.isdir(ho_path):
                    continue

                for trial_folder in sorted(os.listdir(ho_path)):
                    trial_path = os.path.join(ho_path, trial_folder)
                    if not os.path.isdir(trial_path):
                        continue

                    for cam_folder in sorted(os.listdir(trial_path)):
                        cam_path = os.path.join(trial_path, cam_folder)
                        if not os.path.isdir(cam_path):
                            continue

                        rgb_folder = os.path.join(cam_path, "rgb224")
                        if not os.path.isdir(rgb_folder):
                            continue

                        all_jpgs = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
                        if len(all_jpgs) == 0:
                            continue

                        obj_mask_folder = os.path.join(cam_path, "object_mask224")
                        if not os.path.isdir(obj_mask_folder):
                            continue

                        all_masks = sorted(glob.glob(os.path.join(obj_mask_folder, "*.png")))
                        if len(all_masks) == 0:
                            continue

                        filtered_jpgs = []
                        frame_indices = []
                        mask_jpgs = []
                        for path in all_jpgs:
                            frame_id = int(os.path.basename(path).split(".")[0])
                            if (subject, f"{hand_obj_folder}/{trial_folder}/{cam_folder}", frame_id) not in skipped_set:
                                filtered_jpgs.append(path)
                                frame_indices.append(frame_id)
                                mask_jpgs.append(os.path.join(obj_mask_folder, f"{frame_id:06d}.png"))

                        if len(filtered_jpgs) == 0:
                            continue

                        object_key = f"{subject}/{hand_obj_folder}/{trial_folder}/{cam_folder}"
                        self.cached_image_paths[(subject, object_key, self.cameras[0])] = filtered_jpgs
                        self.cached_mask_paths[(subject, object_key, self.cameras[0])] = mask_jpgs

                        cache_key = (subject, object_key)
                        if cache_key not in self.cached_forces:
                            continue

                        valid_force_count = len(self.cached_forces[cache_key]['left'])
                        if valid_force_count == 0:
                            continue

                        for start_idx in range(valid_force_count):
                            samples.append({
                                'participant': subject,
                                'object': object_key,
                                'proc_seqs_path': f"{subject}_{hand_obj_folder}_{trial_folder}_{cam_folder}",
                                'start_idx': start_idx,
                                'length': self.sequence_length,
                                'frame_indices': list(range(valid_force_count)),
                                'total_frame_num': valid_force_count,
                            })

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_images(self, participant, object_name, frame_indices):
        frames = []
        for i in frame_indices:
            img_path = self.cached_images[(participant, object_name, self.cameras[0])][i]
            img = read_image(img_path).float() / 255.0
            frames.append(img)
        return torch.stack(frames), img_path

    def _load_mask(self, participant, object_name, frame_indices):
        frames = []
        for i in frame_indices:
            img_path = self.cached_masks[(participant, object_name, self.cameras[0])][i]
            img = read_image(img_path).float() / 255.0
            frames.append(img)
        return torch.stack(frames)

    def _load_cam_int(self, object_name):
        intrinsics_path = os.path.join(self.images_root, object_name, "cam_intrinsics.txt")
        with open(intrinsics_path, "r") as f:
            line = f.readline().strip()
        values = [float(v) for v in line.split()]

        fx, fy, cx, cy = values[:4]
        K_ego = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return torch.tensor(K_ego)

    def _load_obj_R_T_arti(self, object_name, start_idx, length):
        """
        Loads GT-free object pose/scale plus HAMER hand vertices for this
        sequence. Returns both the raw HAMER output and a copy rigidly
        translated (per-frame centroid match) to align with the dataset's
        own hand-vertex estimate -- HAMER's absolute position isn't metric,
        only its pose/shape are trustworthy, so translation is corrected
        against `vertices_new_cam.npy`.
        """
        hamer_left_path = os.path.join(self.hamer_root, object_name, f'vert_0.0_{start_idx:05d}.npy')
        hamer_right_path = os.path.join(self.hamer_root, object_name, f'vert_1.0_{start_idx:05d}.npy')

        data = np.load(os.path.join(self.images_root, object_name, 'vertices_new_cam.npy'), allow_pickle=True).item()
        left_verts = data['left_hand'][start_idx: start_idx + length]
        right_verts = data['right_hand'][start_idx: start_idx + length]
        world_verts = data['object'][start_idx: start_idx + length]

        if os.path.exists(hamer_left_path):
            hamer_verts_left = np.load(hamer_left_path)
            gt_centroid_0 = left_verts.mean(axis=0)
            pred_centroid_0 = hamer_verts_left.mean(axis=0)
            translation_0 = gt_centroid_0 - pred_centroid_0
            hamer_verts_left_aligned = hamer_verts_left + translation_0
        else:
            hamer_verts_left = None

        if os.path.exists(hamer_right_path):
            hamer_verts_right = np.load(hamer_right_path)
            gt_centroid_1 = right_verts.mean(axis=0)
            pred_centroid_1 = hamer_verts_right.mean(axis=0)
            translation_1 = gt_centroid_1 - pred_centroid_1
            hamer_verts_right_aligned = hamer_verts_right + translation_1
        else:
            hamer_verts_right = None

        if hamer_verts_left is None:
            hamer_verts_left = np.zeros((1, 778, 3))
            hamer_verts_left_aligned = np.zeros((1, 778, 3))
        if hamer_verts_right is None:
            hamer_verts_right = np.zeros((1, 778, 3))
            hamer_verts_right_aligned = np.zeros((1, 778, 3))

        scale = np.load(os.path.join(self.rot_trans_scale_root, object_name, 'scale_local_cam_normalized.npy'))[start_idx: start_idx + length]
        R = np.load(os.path.join(self.rot_trans_scale_root, object_name, 'rot_local_cam_normalized.npy'))[start_idx: start_idx + length]
        T = np.load(os.path.join(self.rot_trans_scale_root, object_name, 'trans_local_cam_normalized.npy'))[start_idx: start_idx + length]

        return (
            torch.tensor(world_verts, dtype=torch.float32),
            torch.tensor(left_verts, dtype=torch.float32),
            torch.tensor(right_verts, dtype=torch.float32),
            torch.tensor(scale, dtype=torch.float32),
            torch.tensor(hamer_verts_left, dtype=torch.float32),
            torch.tensor(hamer_verts_right, dtype=torch.float32),
            torch.tensor(hamer_verts_left_aligned, dtype=torch.float32),
            torch.tensor(hamer_verts_right_aligned, dtype=torch.float32),
            torch.tensor(R, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32),
        )

    def _load_contacts(self, participant, object_name, start_idx, length):
        contact_data = {}
        force_data = {}
        force_vector = {}

        contacts_dict = self.cached_contacts.get((participant, object_name), {})
        forces_dict = self.cached_forces.get((participant, object_name), {})

        for suffix in ['left', 'right', 'object']:
            contact_path = contacts_dict.get(suffix, None)
            if contact_path is None or not os.path.isfile(contact_path):
                print(f"Warning: contact path {contact_path} not found, skipping")
                continue

            arr_contact = np.load(contact_path, mmap_mode='r')
            total_len = len(arr_contact)

            valid_frame_ids = [i for i in range(len(arr_contact)) if (participant, object_name, i) not in self.to_skip]
            valid_frame_ids = np.array(valid_frame_ids, dtype=int)
            arr_contact_clean = arr_contact[valid_frame_ids]

            contact_data[suffix] = torch.from_numpy(arr_contact_clean[start_idx:start_idx + length]).float()

            force_files = forces_dict.get(suffix, [])
            if not force_files:
                continue
            all_forces = np.stack([np.load(f, mmap_mode='r') for f in force_files], axis=0)
            all_forces = all_forces[valid_frame_ids]

            all_forces_masked = mask_forces_with_contacts(all_forces, arr_contact_clean, force_files, contact_path, total_len)
            all_forces_masked = torch.from_numpy(all_forces_masked).float()

            # --- Force magnitude normalization to [0, 1] using global max magnitudes ---
            magnitudes = torch.norm(all_forces_masked, dim=2)  # [B, N_vertices]
            max_mag = config.H2O_FORCE_MAX_MAGNITUDE[suffix]
            norm_magnitude = torch.clamp(magnitudes / max_mag, 0.0, 1.0)

            # --- Force vector normalization to [-1, 1] ---
            direction = all_forces_masked / (magnitudes.unsqueeze(-1) + 1e-8)
            direction = torch.clamp(direction, -1.0, 1.0)

            force_data[suffix] = norm_magnitude[start_idx:start_idx + length]
            force_vector[suffix] = direction[start_idx:start_idx + length]

        return contact_data, force_data, force_vector

    def _load_articulated_mesh(self, object_name):
        """
        Loads the object mesh for this sequence directly from its own
        frame-0 mesh file (H2O has no shared per-category template like
        ARCTIC does).
        """
        mesh_whole = trimesh.load(os.path.join(self.images_root, object_name, 'object_frame0_new.obj'), process=False)
        verts_whole = np.array(mesh_whole.vertices)

        coords = torch.from_numpy(verts_whole).float()
        center = coords.mean(dim=0, keepdim=True)
        scale = coords.abs().max()
        coords = (coords - center) / scale

        edges = np.array(mesh_whole.edges_unique)

        return coords, edges

    def __getitem__(self, idx):
        sample = self.samples[idx]
        participant = sample['participant']
        object_name = sample['object']
        start_idx = sample['start_idx']
        length = sample['length']
        frame_indices = sample['frame_indices'][start_idx:start_idx + length]

        images, img_path = self._load_images(participant, object_name, frame_indices)
        obj_mask = self._load_mask(participant, object_name, frame_indices)
        K_ego = self._load_cam_int(object_name)
        (world_verts, left_verts, right_verts, scale,
         hamer_left_orig, hamer_right_orig,
         hamer_left_aligned, hamer_right_aligned,
         R, T) = self._load_obj_R_T_arti(object_name, start_idx, length)
        contacts, forces, force_vector = self._load_contacts(participant, object_name, start_idx, length)

        K_ego = K_ego.unsqueeze(0).expand(self.sequence_length, -1, -1)
        mesh, edges = self._load_articulated_mesh(object_name)

        return {
            'images': images,
            'img_path': img_path,
            'obj_segmentation': obj_mask,
            'scale': scale,
            'contacts': contacts,
            'forces': forces,
            'force_vector': force_vector,
            'obj_vert_template': mesh,
            'obj_edges_template': edges,
            'participant': participant,
            'object': object_name,
            'proc_seqs_path': sample['proc_seqs_path'],
            'K_ego': K_ego,
            'R': R,
            'T': T,
            's': scale,
            'world_verts': world_verts,
            'left_verts': left_verts,
            'right_verts': right_verts,
            'hamer_left_orig': hamer_left_orig,
            'hamer_right_orig': hamer_right_orig,
            'hamer_left_aligned': hamer_left_aligned,
            'hamer_right_aligned': hamer_right_aligned,
        }