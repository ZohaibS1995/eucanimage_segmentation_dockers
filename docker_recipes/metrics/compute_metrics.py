#!/usr/bin/env python3

from __future__ import division
import io
import os
import json
import pandas as pd
import numpy as np
import tarfile
import nibabel as nib
from argparse import ArgumentParser
import JSON_templates
import torch
from scipy.spatial.distance import cdist
from monai.metrics import (
    DiceMetric, SurfaceDiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric,
)
from monai.transforms import AsDiscrete

# Initialize MONAI transforms and metrics
as_discrete = AsDiscrete(threshold=0.5)
def _load_mask(fname: str, label: int = 1) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load NIfTI mask for specific label → boolean array + voxel spacing."""
    img = nib.load(fname)
    data = img.get_fdata() == label  # Only extract specified label
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms


def _tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert numpy mask to PyTorch tensor with proper dimensions."""
    t = torch.from_numpy(mask.astype(np.uint8))
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t.unsqueeze(0).unsqueeze(0)


def _modified_hausdorff(a: np.ndarray, b: np.ndarray,
                        spacing: tuple[float, float, float]) -> float:
    """Compute modified Hausdorff distance."""
    pts_a = np.argwhere(a) * spacing
    pts_b = np.argwhere(b) * spacing
    if pts_a.size == 0 or pts_b.size == 0:
        return np.nan
    d = cdist(pts_a, pts_b)
    return max(d.min(1).mean(), d.min(0).mean())


def extract_tarfile(input_file, extract_path):
    """
    Extracts a tar file to the specified path.

    :param input_file: Path to the tar file
    :param extract_path: Directory to extract files into
    :return: List of extracted file paths
    """
    with tarfile.open(input_file, 'r:gz') as tar:
        tar.extractall(path=extract_path)
        return [os.path.join(extract_path, member.name) for member in tar.getmembers() if member.isfile()]


def compute_comprehensive_metrics(pred_files: list, gt_files: list, tolerance_mm: float = 1.0):
    """
    Compute comprehensive segmentation metrics using MONAI, focusing on label 1 only.

    :param pred_files: List of prediction file paths
    :param gt_files: List of ground truth file paths
    :param tolerance_mm: Tolerance in mm for surface dice metric
    :return: Dictionary with metric statistics
    """
    keys = ("dice", "jaccard", "surface_dice", "hausdorff",
            "hausdorff95", "assd", "nsd", "modified_hausdorff")
    scores = {k: [] for k in keys}

    # Initialize MONAI metrics
    dice_m = DiceMetric(include_background=False)
    assd_m = SurfaceDistanceMetric(include_background=False, symmetric=True)
    hd_m = HausdorffDistanceMetric(include_background=False)
    hd95_m = HausdorffDistanceMetric(include_background=False, percentile=95.0)

    for pf, gf in zip(pred_files, gt_files):
        print(f"Processing: {os.path.basename(pf)} vs {os.path.basename(gf)}")

        # Load masks for label 1 only
        pred, sp = _load_mask(pf, label=1)
        gt, _ = _load_mask(gf, label=1)

        # Convert to tensors
        tp, tg = as_discrete(_tensor(pred)), as_discrete(_tensor(gt))

        # Region-based metrics
        dice_val = float(dice_m(tp, tg).item())
        scores["dice"].append(dice_val)

        # Jaccard (IoU)
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        jaccard_val = inter / (union + 1e-8) if union > 0 else 0.0
        scores["jaccard"].append(jaccard_val)

        # Surface-based metrics
        if pred.sum() > 0 and gt.sum() > 0:  # Only compute if both masks have foreground
            try:
                # Surface Dice / NSD
                thr_vox = tolerance_mm / float(np.mean(sp))
                sdc = SurfaceDiceMetric(class_thresholds=[thr_vox], include_background=False)
                sd_val = float(sdc(tp, tg).item())
                scores["surface_dice"].append(sd_val)
                scores["nsd"].append(sd_val)

                # Hausdorff distances
                hd_val = float(hd_m(tp, tg, spacing=sp).item())
                hd95_val = float(hd95_m(tp, tg, spacing=sp).item())
                assd_val = float(assd_m(tp, tg, spacing=sp).item())

                scores["hausdorff"].append(hd_val)
                scores["hausdorff95"].append(hd95_val)
                scores["assd"].append(assd_val)

                # Modified Hausdorff
                mod_hd = _modified_hausdorff(pred, gt, sp)
                scores["modified_hausdorff"].append(mod_hd)

            except Exception as e:
                print(f"Warning: Could not compute surface metrics for {os.path.basename(pf)}: {e}")
                # Append NaN for failed surface metrics
                for key in ["surface_dice", "nsd", "hausdorff", "hausdorff95", "assd", "modified_hausdorff"]:
                    scores[key].append(np.nan)
        else:
            print(f"Warning: Empty masks detected for {os.path.basename(pf)}, skipping surface metrics")
            # Append NaN for empty masks
            for key in ["surface_dice", "nsd", "hausdorff", "hausdorff95", "assd", "modified_hausdorff"]:
                scores[key].append(np.nan)

    # Compute statistics
    return {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
            for k, v in scores.items()}


def main(args):
    # input parameters
    input_file = args.input
    goldstandard_dir = args.goldstandard_dir
    challenges_ids = args.challenges_ids
    event = args.event_id
    participant_id = args.participant_id
    community = args.community_id
    outdir = args.outdir

    print(f"INFO: participant input file {input_file}")
    print(f"INFO: Selected challenge(s) {challenges_ids}")

    ### -------------------------------------------------------------------------------------------------------
    # Untar input file (single tar file of multiple nifti files)
    # Directory were the .gz files are going to be extracted
    untar_dir = os.path.join(os.path.dirname(input_file), 'files')
    print(f"INFO: Untar dir {untar_dir}.")

    # Check if the directory already exists
    if not os.path.exists(untar_dir):
        # Create the directory
        os.makedirs(untar_dir)
        extracted_files = extract_tarfile(input_file, untar_dir)
    else:
        print(f"INFO: Directory {untar_dir} already exists.")
        extracted_files = [os.path.join(untar_dir, f) for f in os.listdir(untar_dir) if
                           os.path.isfile(os.path.join(untar_dir, f))]

    print(f"INFO: Extracting files to {untar_dir}. Extracted files: {extracted_files}")
    ### -------------------------------------------------------------------------------------------------------

    # Assuring the output path does exist
    if not os.path.exists(os.path.dirname(outdir)):
        try:
            os.makedirs(os.path.dirname(outdir))
            with open(outdir, mode="a"):
                pass
        except OSError as exc:
            print("OS error: {0}".format(exc) +
                  "\nCould not create output path: " + outdir)

    compute_metrics(input_file, goldstandard_dir,
                    challenges_ids, participant_id, community, event, outdir)


def compute_metrics(input_file, goldstandard_dir, challenge, participant_id,
                    community, event, outdir):
    untar_dir = os.path.join(os.path.dirname(input_file), 'files')
    gt_path = goldstandard_dir

    # Get and sort seg files
    seg_files = sorted([f for f in os.listdir(untar_dir)
                          if f.startswith('seg_') and f.endswith('.nii.gz')])
    seg_files = [os.path.join(untar_dir, f) for f in seg_files]
    print(f"Sorted seg files: {seg_files}")

    gt_files = sorted([f for f in os.listdir(gt_path)
                       if f.startswith('gt_') and f.endswith('.nii.gz')])

    print(f"INFO: Found GT files: {gt_files}")

    gt_files = [os.path.join(gt_path, f) for f in gt_files]
    print(f"Sorted GT files: {gt_files}")

    assert len(seg_files) == len(gt_files), \
        f"ERROR: Number of submitted segmentation files {len(seg_files)} does not match number of ground truth files {len(gt_files)}!"

    # define array that will hold the full set of assessment datasets
    all_assessments = []

    # ID prefix for assessment objects
    if isinstance(challenge, list):
        challenge = challenge[0]
    base_id = f"{community}:{event}_{challenge}_{participant_id}:"

    # Compute comprehensive metrics for label 1 only
    print("Computing comprehensive metrics for label 1...")
    metrics_summary = compute_comprehensive_metrics(seg_files, gt_files, tolerance_mm=1.0)

    print("Computed metrics:")
    for metric, stats in metrics_summary.items():
        print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Create assessment objects for each metric
    for metric_name, stats in metrics_summary.items():
        object_id = base_id + f"{metric_name}_label1"
        assessment_object = JSON_templates.write_assessment_dataset(
            object_id, community, challenge, participant_id,
            f"{metric_name}_label1", stats['mean'], stats['std'])
        all_assessments.append(assessment_object)

    # Write all assessments to JSON file
    with io.open(outdir, mode='w', encoding="utf-8") as f:
        jdata = json.dumps(all_assessments, sort_keys=True,
                           indent=4, separators=(',', ': '))
        f.write(jdata)

    print(f"Assessment results written to: {outdir}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="List of execution workflow output files", required=True)
    parser.add_argument("-c", "--challenges_ids", nargs='+',
                        help="List of challenges ids selected by the user, separated by spaces", required=True)
    parser.add_argument("-g", "--goldstandard_dir",
                        help="dir that contains gold standard datasets for current challenge", required=True)
    parser.add_argument("-p", "--participant_id",
                        help="name of the tool used for prediction", required=True)
    parser.add_argument("-com", "--community_id",
                        help="name/id of benchmarking community", required=True)
    parser.add_argument("-e", "--event_id",
                        help="name/id of benchmarking event", required=True)
    parser.add_argument("-o", "--outdir",
                        help="output path where assessment JSON files will be written", required=True)

    args = parser.parse_args()
    main(args)