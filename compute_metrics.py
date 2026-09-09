"""
Contact and force metrics for EgoPHI predictions, computed purely from saved
.pt files -- no model, dataset, or GPU involved. Reproduces the methodology
of the original `eccv_COMPARE_ARCTIC_*.py` / `eccv_compare_H2O_*.py` scripts:

  - Hand contact/force pool left and right together into a single "hand"
    group (matching Table 2's "hand contact"/"hand force" columns); object
    is reported separately.
  - Contact: precision / recall / F1 / IoU (sklearn), computed per frame and
    averaged. Hands use a fixed threshold (HAND_CONTACT_THRESHOLD); the
    object uses a per-frame dynamic threshold
    `min(pred) + OBJECT_DYNAMIC_FACTOR * (max(pred) - min(pred))`, since a
    single fixed threshold doesn't transfer well across very differently
    shaped/scaled objects.
  - Force: magnitude predictions and ground truth are both scaled by
    FORCE_SCALE to get Newtons, then scored with Masked MAE / Masked RMSE
    (within the ground-truth contact mask only) and volumetric IoU (over all
    vertices, unmasked).
  - Frames with no ground-truth contact at all are skipped, as in the
    original scripts (an empty GT mask makes precision/recall/IoU
    ill-defined, and force error meaningless).

Expects two directory layouts of the same shape -- predictions from
evaluate_ARCTIC.py/evaluate_H2O.py, and ground truth from wherever you've
saved it in the same layout (see their commented-out "Ground truth" blocks
for how to produce one):

    <predictions_dir>/<sequence_name>/{left,right,obj}_{contact,force_mag}_pred_<frame_id>.pt
    <gt_dir>/<sequence_name>/gt_{left,right,obj}_{contact,force_mag}_<frame_id>.pt

Usage:
    python compute_metrics.py --predictions-dir /path/to/evaluation_results/arctic --gt-dir /path/to/gt
    python compute_metrics.py --predictions-dir /path/to/evaluation_results/h2o    --gt-dir /path/to/gt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, jaccard_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

HAND_CONTACT_THRESHOLD = 0.3
OBJECT_DYNAMIC_FACTOR = 0.55
FORCE_SCALE = 100.0  # normalized [0, 1] force_mag * FORCE_SCALE = Newtons
FORCE_CONTACT_MASK_THRESHOLD = 0.1  # GT contact value above which a vertex counts as "in contact" for masked force metrics
HAND_SIDES = ("left", "right")


def _load(path):
    return torch.load(path, weights_only=True).flatten().float().detach().numpy()


def _iter_frames(predictions_dir, gt_dir, side, kind):
    """Yields (pred_array, gt_array) for every prediction file with a matching ground-truth file."""
    predictions_dir, gt_dir = Path(predictions_dir), Path(gt_dir)
    for seq_dir in sorted(predictions_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        gt_seq_dir = gt_dir / seq_dir.name
        if not gt_seq_dir.is_dir():
            continue
        for pred_path in sorted(seq_dir.glob(f"{side}_{kind}_pred_*.pt")):
            frame_id = pred_path.stem.split("_")[-1]
            gt_path = gt_seq_dir / f"gt_{side}_{kind}_{frame_id}.pt"
            if gt_path.is_file():
                yield _load(pred_path), _load(gt_path)


def _contact_scores(pred, gt_bin, threshold):
    pred_bin = (pred > threshold).astype(np.uint8)
    return (
        precision_score(gt_bin, pred_bin, zero_division=0),
        recall_score(gt_bin, pred_bin, zero_division=0),
        f1_score(gt_bin, pred_bin, zero_division=0),
        jaccard_score(gt_bin, pred_bin, zero_division=0),
    )


def contact_metrics(predictions_dir, gt_dir, sides, threshold=HAND_CONTACT_THRESHOLD, dynamic=False):
    """
    Per-frame precision/recall/F1/IoU, averaged over frames (and over
    `sides`, e.g. both hands pooled into one set of numbers). `dynamic=True`
    ignores `threshold` and instead uses the per-frame
    min+OBJECT_DYNAMIC_FACTOR*(max-min) threshold (for the object).
    """
    rows = []
    for side in sides:
        for pred, gt in _iter_frames(predictions_dir, gt_dir, side, "contact"):
            gt_bin = (gt > 0.5).astype(np.uint8)
            if gt_bin.sum() == 0:
                continue
            frame_threshold = pred.min() + OBJECT_DYNAMIC_FACTOR * (pred.max() - pred.min()) if dynamic else threshold
            rows.append(_contact_scores(pred, gt_bin, frame_threshold))

    if not rows:
        return 0, None
    precision, recall, f1, iou = np.array(rows).mean(axis=0)
    return len(rows), {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}


def _masked_mae_rmse(pred, gt, mask):
    if mask.sum() == 0:
        return None, None
    pred_m, gt_m = pred[mask], gt[mask]
    finite = np.isfinite(pred_m) & np.isfinite(gt_m)
    pred_m, gt_m = pred_m[finite], gt_m[finite]
    if pred_m.size == 0:
        return None, None
    return mean_absolute_error(gt_m, pred_m), np.sqrt(mean_squared_error(gt_m, pred_m))


def _volumetric_iou(pred, gt):
    finite = np.isfinite(pred) & np.isfinite(gt)
    pred, gt = pred[finite], gt[finite]
    if pred.size == 0:
        return None
    union = np.maximum(pred, gt).sum()
    return np.minimum(pred, gt).sum() / (union + 1e-8) if union > 0 else 0.0


def force_metrics(predictions_dir, gt_dir, sides):
    """
    Masked MAE/RMSE (Newtons, within the GT contact mask) and volumetric IoU
    (unmasked), averaged per frame over `sides`. Predictions and ground
    truth are both scaled by FORCE_SCALE to get Newtons.
    """
    mae_list, rmse_list, viou_list = [], [], []
    for side in sides:
        force_pairs = list(_iter_frames(predictions_dir, gt_dir, side, "force_mag"))
        contact_pairs = list(_iter_frames(predictions_dir, gt_dir, side, "contact"))
        for (pred, gt), (_, gt_contact) in zip(force_pairs, contact_pairs):
            contact_mask = gt_contact > FORCE_CONTACT_MASK_THRESHOLD
            if contact_mask.sum() == 0:
                continue
            pred_n, gt_n = pred * FORCE_SCALE, gt * FORCE_SCALE

            mae, rmse = _masked_mae_rmse(pred_n, gt_n, contact_mask)
            viou = _volumetric_iou(pred_n, gt_n)
            if mae is not None:
                mae_list.append(mae)
                rmse_list.append(rmse)
            if viou is not None:
                viou_list.append(viou)

    if not mae_list:
        return 0, None
    return len(mae_list), {
        'mae': np.mean(mae_list), 'rmse': np.mean(rmse_list), 'vol_iou': np.mean(viou_list),
    }


def print_contact_table(title, n_frames, m):
    print(f"\n--- {title} contact ({n_frames} frames) ---")
    if m is None:
        print("  no matching prediction/GT frames found")
        return
    print(f"  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}  IoU: {m['iou']:.4f}")


def print_force_table(title, n_frames, m):
    print(f"\n--- {title} force ({n_frames} frames) ---")
    if m is None:
        print("  no matching prediction/GT frames found")
        return
    print(f"  MAE: {m['mae']:.4f} N   RMSE: {m['rmse']:.4f} N   Volumetric IoU: {m['vol_iou']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", type=Path, required=True,
                         help="e.g. os.path.join(config.PROJECT_ROOT, 'evaluation_results', 'arctic')")
    parser.add_argument("--gt-dir", type=Path, required=True,
                         help="Same per-sequence layout, with gt_<side>_<kind>_<frame_id>.pt files.")
    args = parser.parse_args()

    print(f"Predictions: {args.predictions_dir}")
    print(f"Ground truth: {args.gt_dir}")

    n, m = contact_metrics(args.predictions_dir, args.gt_dir, HAND_SIDES, threshold=HAND_CONTACT_THRESHOLD)
    print_contact_table(f"Hand (threshold={HAND_CONTACT_THRESHOLD})", n, m)
    n, m = contact_metrics(args.predictions_dir, args.gt_dir, ("obj",), dynamic=True)
    print_contact_table("Object (dynamic per-frame threshold)", n, m)

    n, m = force_metrics(args.predictions_dir, args.gt_dir, HAND_SIDES)
    print_force_table("Hand", n, m)
    n, m = force_metrics(args.predictions_dir, args.gt_dir, ("obj",))
    print_force_table("Object", n, m)


if __name__ == '__main__':
    main()
