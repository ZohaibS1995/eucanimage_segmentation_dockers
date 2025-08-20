from __future__ import annotations

import io
import os
import sys
import json
import tarfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import torch
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

from monai.metrics import (
    DiceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
    HausdorffDistanceMetric,
)
from monai.transforms import AsDiscrete

import JSON_templates  # provided by the evaluation environment


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

as_discrete = AsDiscrete(threshold=0.5)


def _load_mask(fname: str, label: int = 1) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load NIfTI mask for a specific label -> boolean array + voxel spacing."""
    img = nib.load(fname)
    data = (img.get_fdata() == label)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data.astype(bool, copy=False), zooms


def _to_tensor(mask: np.ndarray) -> torch.Tensor:
    """
    Convert numpy mask to float32 tensor with MONAI/torch expected shape.
    Returns (B, C, H, W) for 2D, or (B, C, D, H, W) for 3D.
    """
    t = torch.from_numpy(mask.astype(np.float32, copy=False))
    if t.ndim == 2:        # (H, W) -> (1,1,H,W)
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:      # (D, H, W) -> (1,1,D,H,W)
        t = t.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported mask dims {t.shape}")
    return t


def _surface_points(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """
    Return physical-coordinate surface points of a binary mask.
    We define surface as mask voxels that erode away with a 3x3 (2D) or 3x3x3 (3D) structuring element.
    """
    if mask.ndim == 2:
        structure = np.ones((3, 3), dtype=bool)
    elif mask.ndim == 3:
        structure = np.ones((3, 3, 3), dtype=bool)
    else:
        raise ValueError(f"Unsupported mask dims {mask.shape}")

    if not mask.any():
        return np.empty((0, mask.ndim), dtype=np.float32)

    eroded = binary_erosion(mask, structure=structure, border_value=0)
    surf = mask & ~eroded
    idx = np.argwhere(surf)  # integer voxel indices
    if idx.size == 0:
        return np.empty((0, mask.ndim), dtype=np.float32)

    sp_arr = np.asarray(spacing[:mask.ndim], dtype=np.float32)
    return idx.astype(np.float32, copy=False) * sp_arr  # physical coordinates


def _modified_hausdorff(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """
    Memory-safe Modified Hausdorff distance (MHD) using KD-Trees on SURFACE voxels.
    Returns 0.0 if either mask has no foreground (as per "no NaNs" guarantee).

    MHD(A,B) = max( mean_{x in A} min_{y in B} ||x - y|| , mean_{y in B} min_{x in A} ||x - y|| )
    """
    # Early outs for empty masks
    if not a.any() or not b.any():
        return 0.0

    pts_a = _surface_points(a, spacing)
    pts_b = _surface_points(b, spacing)

    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        # If either surface is empty (e.g., single-voxel object erodes away), fall back to all foreground points
        # but still do KD-Tree, not cdist, to remain memory-safe.
        pts_a = np.argwhere(a).astype(np.float32, copy=False)
        pts_b = np.argwhere(b).astype(np.float32, copy=False)
        sp_arr = np.asarray(spacing[:a.ndim], dtype=np.float32)
        pts_a *= sp_arr
        pts_b *= sp_arr
        if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
            return 0.0

    try:
        tree_b = cKDTree(pts_b)
        d_ab, _ = tree_b.query(pts_a, k=1)  # nearest neighbor distances from A->B
        mean_ab = float(np.mean(d_ab)) if d_ab.size else 0.0

        tree_a = cKDTree(pts_a)
        d_ba, _ = tree_a.query(pts_b, k=1)  # nearest neighbor distances from B->A
        mean_ba = float(np.mean(d_ba)) if d_ba.size else 0.0

        return float(max(mean_ab, mean_ba))
    except Exception:
        # Fail-safe: if KD-Tree fails for any reason, return 0.0 per guarantees.
        return 0.0


def _extract_tar_if_needed(input_path: Path) -> List[Path]:
    """
    If input is a .tar.gz, extract to sibling 'files' dir and return extracted file paths.
    If input is a directory, return contained files.
    Otherwise, assume it's already a single file and return as [input_path].
    """
    if input_path.is_dir():
        return [p for p in input_path.iterdir() if p.is_file()]

    suffixes = input_path.suffixes
    if len(suffixes) >= 2 and suffixes[-2:] == ['.tar', '.gz']:
        out_dir = input_path.parent / 'files'
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(input_path, 'r:gz') as tar:
            # NOTE: This trusts the tarball source as in the original script.
            tar.extractall(path=out_dir)
        return [p for p in out_dir.iterdir() if p.is_file()]

    return [input_path]


def _safe_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Convert list to numeric array, replace NaN/Inf with 0, and compute mean/std.
    If list is empty, returns (0.0, 0.0).
    """
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(arr.mean()), float(arr.std(ddof=0))


# -----------------------------------------------------------------------------#
# Core
# -----------------------------------------------------------------------------#

def compute_segmentation_metrics(
    pred_files: List[Path],
    gt_files: List[Path],
    tolerance_mm: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-case metrics and return aggregated mean/std.
    Guarantees 0.0 for any undefined/failed values.
    """
    keys = ("dice", "jaccard", "surface_dice", "hausdorff", "hausdorff95", "assd", "modified_hausdorff")
    scores = {k: [] for k in keys}

    # Use fresh metric instances per case to avoid accidental accumulation.
    for pf, gf in zip(pred_files, gt_files):
        # Load
        pred, sp = _load_mask(str(pf), label=1)  # change label here if needed
        gt, _ = _load_mask(str(gf), label=1)

        # Tensors (discrete thresholding keeps them 0/1)
        tp = as_discrete(_to_tensor(pred))
        tg = as_discrete(_to_tensor(gt))

        # Dice
        try:
            d_metric = DiceMetric(include_background=False, reduction="mean")
            dice_val = float(d_metric(tp, tg).item())
        except Exception:
            dice_val = 0.0
        scores["dice"].append(dice_val)

        # Jaccard (IoU)
        inter = float(np.logical_and(pred, gt).sum())
        union = float(np.logical_or(pred, gt).sum())
        jaccard_val = (inter / union) if union > 0.0 else 0.0
        scores["jaccard"].append(float(jaccard_val))

        # Surface-based metrics (only meaningful if both have foreground)
        if pred.any() and gt.any():
            try:
                # Surface Dice (Normalized Surface Dice) threshold in voxels (approx via mean spacing)
                thr_vox = float(tolerance_mm / max(1e-8, float(np.mean(sp))))
                sdc = SurfaceDiceMetric(class_thresholds=[thr_vox], include_background=False, reduction="mean")
                sd_val = float(sdc(tp, tg).item())
            except Exception:
                sd_val = 0.0
            scores["surface_dice"].append(sd_val)

            # Hausdorff distances & ASSD
            try:
                hd = HausdorffDistanceMetric(include_background=False, reduction="mean")            # 100%
                hd95 = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
                assd = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")

                hd_val = float(hd(tp, tg, spacing=sp).item())
                hd95_val = float(hd95(tp, tg, spacing=sp).item())
                assd_val = float(assd(tp, tg, spacing=sp).item())
            except Exception:
                hd_val = hd95_val = assd_val = 0.0

            scores["hausdorff"].append(hd_val)
            scores["hausdorff95"].append(hd95_val)
            scores["assd"].append(assd_val)

            # Modified Hausdorff (memory-safe: KD-Tree on surfaces)
            mh_val = _modified_hausdorff(pred, gt, sp)
            scores["modified_hausdorff"].append(float(mh_val))
        else:
            # No foreground in either/both -> set surface-based metrics to 0
            for k in ("surface_dice", "hausdorff", "hausdorff95", "assd", "modified_hausdorff"):
                scores[k].append(0.0)

    # Aggregate
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in scores.items():
        mean, std = _safe_mean_std(vals)
        out[k] = {"mean": mean, "std": std}
    return out


def main(cfg):
    # Accept either a dir or a file path. If a dir contains "segmentations.tar.gz", use that.
    inp = Path(cfg.input)
    if inp.is_dir():
        tar_candidate = inp / "segmentations.tar.gz"
        if tar_candidate.exists():
            inp = tar_candidate

    gt_dir = Path(cfg.goldstandard_dir)

    if not gt_dir.is_dir():
        sys.exit(f"ERROR: Ground-truth directory '{gt_dir}' does not exist or is not a directory.")

    # Collect prediction files (tar.gz or directory)
    extracted = _extract_tar_if_needed(inp)
    pred_files = sorted(
        [Path(p) for p in extracted if Path(p).name.startswith("seg_") and Path(p).suffixes[-2:] == ['.nii', '.gz']]
    )
    if not pred_files:
        sys.exit(f"ERROR: No prediction NIfTI files named 'seg_*.nii.gz' found in '{inp}'.")

    # Collect GT files and map by id
    gt_all = sorted([p for p in gt_dir.iterdir() if p.is_file() and p.name.startswith("gt_") and p.suffixes[-2:] == ['.nii', '.gz']])
    gt_by_id = {p.stem.replace("gt_", "").replace(".nii", ""): p for p in gt_all}

    # Pair predictions with matching GT by id (seg_X <-> gt_X)
    paired_pred: List[Path] = []
    paired_gt: List[Path] = []
    missing: List[str] = []

    for pf in pred_files:
        pid = pf.stem.replace("seg_", "").replace(".nii", "")
        expected_gt = gt_by_id.get(pid)
        if expected_gt is None:
            missing.append(pid)
        else:
            paired_pred.append(pf)
            paired_gt.append(expected_gt)

    if missing:
        sys.exit(f"ERROR: Missing GT for {len(missing)} prediction(s): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    if not paired_pred:
        sys.exit("ERROR: No matched prediction/GT pairs found.")

    # Compute metrics
    metrics_summary = compute_segmentation_metrics(paired_pred, paired_gt, tolerance_mm=1.0)

    # Prepare assessment objects (same JSON template style as classification)
    challenge = cfg.challenges_ids[0] if isinstance(cfg.challenges_ids, list) else cfg.challenges_ids
    base_id = f"{cfg.community_id}:{cfg.event_id}_{challenge}_{cfg.participant_id}:"

    assessments: List[dict] = []
    for metric_name, stats in metrics_summary.items():
        mean_val = float(stats.get("mean", 0.0) or 0.0)
        std_val = float(stats.get("std", 0.0) or 0.0)
        assessments.append(
            JSON_templates.write_assessment_dataset(
                base_id + metric_name,
                cfg.community_id,
                challenge,
                cfg.participant_id,
                metric_name,
                mean_val,
                std_val,
            )
        )

    # Robust output path (like classification script)
    out_json_path = Path(cfg.outdir)
    if not out_json_path.is_absolute():
        out_json_path = Path.cwd() / out_json_path
    if out_json_path.parent != Path('.'):
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

    with io.open(out_json_path, mode="w", encoding="utf-8") as fp:
        json.dump(assessments, fp, indent=4, sort_keys=True, separators=(",", ": "))

    print(f"INFO: Wrote metrics JSON → {out_json_path}")
    sys.exit(0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute binary segmentation metrics for EuCanImage challenges.")
    parser.add_argument("-i", "--input", required=True,
                        help="Tarball (.tar.gz) of NIfTI files named seg_*.nii.gz, or a directory containing them.")
    parser.add_argument("-g", "--goldstandard_dir", required=True,
                        help="Directory containing ground-truth NIfTI masks named gt_*.nii.gz.")
    parser.add_argument("-c", "--challenges_ids", nargs="+", required=True,
                        help="Challenge id(s), space-separated.")
    parser.add_argument("-p", "--participant_id", required=True,
                        help="Tool / model id (participant).")
    parser.add_argument("-com", "--community_id", required=True,
                        help="Benchmarking community id (e.g. 'EuCanImage').")
    parser.add_argument("-e", "--event_id", required=True,
                        help="Benchmarking event id.")
    parser.add_argument("-o", "--outdir", required=True,
                        help="Path to metrics JSON (other artefacts share the same basename).")

    args = parser.parse_args()
    main(args)
